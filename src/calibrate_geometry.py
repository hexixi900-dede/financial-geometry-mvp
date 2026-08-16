from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from geometry_core import draw_debug_overlay, token_from_dict


FEATURE_NAMES = [
    "raw_axis_position",
    "target_y_fraction",
    "tick_step_fraction",
    "pixel_step_fraction",
    "axis_r_squared",
    "axis_residual_fraction",
    "geometry_score",
    "bar_width_fraction",
    "bar_height_fraction",
    "bar_fill_ratio",
    "bar_base_distance_fraction",
    "line_both_sides",
    "line_fit_residual_fraction",
    "line_local_columns_fraction",
    "is_line",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_score(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def assign_chart_splits(rows: list[dict[str, Any]], seed: int, test_fraction: float) -> dict[str, str]:
    types_by_chart: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        types_by_chart[str(row["chart_id"])][str(row["geometry"]["chart_type"])] += 1
    charts_by_type: dict[str, list[str]] = defaultdict(list)
    for chart_id, counts in types_by_chart.items():
        charts_by_type[counts.most_common(1)[0][0]].append(chart_id)
    split: dict[str, str] = {}
    for chart_type, chart_ids in sorted(charts_by_type.items()):
        ordered = sorted(chart_ids, key=lambda chart_id: stable_score(chart_id, seed))
        if len(ordered) <= 1:
            test_count = 0
        else:
            test_count = max(1, int(round(test_fraction * len(ordered))))
            test_count = min(test_count, len(ordered) - 1)
        for index, chart_id in enumerate(ordered):
            split[chart_id] = "test" if index < test_count else "train"
    return split


def feature_vector(row: dict[str, Any]) -> np.ndarray:
    axis = row["axis"]
    geometry = row["geometry"]
    left, top, right, bottom = map(float, axis["plot_bbox"])
    plot_width = max(1.0, right - left)
    plot_height = max(1.0, bottom - top)
    axis_span = max(float(axis["axis_span"]), 1e-12)
    raw = float(row["raw_geometry_value"])
    raw_position = (raw - float(axis["axis_min_display"])) / axis_span
    target_y_fraction = (float(geometry["target_y"]) - top) / plot_height
    bbox = geometry.get("bbox")
    is_line = geometry["chart_type"] == "line"
    features = [
        raw_position,
        target_y_fraction,
        float(axis["median_value_step"]) / axis_span,
        float(axis["median_pixel_step"]) / plot_height,
        float(axis["r_squared"]),
        float(axis["max_residual_pixels"]) / plot_height,
        float(geometry["score"]),
        0.0 if bbox is None else float(bbox[2]) / plot_width,
        0.0 if bbox is None else float(bbox[3]) / plot_height,
        float(geometry.get("fill_ratio", 0.0)),
        float(geometry.get("base_distance_pixels", 0.0)) / plot_height,
        float(geometry.get("both_sides", 0.0)),
        float(geometry.get("local_fit_residual_pixels", 0.0)) / plot_height,
        float(geometry.get("local_columns", 0.0)) / plot_width,
        float(is_line),
    ]
    return np.asarray(features, dtype=float)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def fit_huber_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    delta: float,
    max_iterations: int = 60,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = fit_ridge(x, y, alpha)
    for _ in range(max_iterations):
        residual = y - design @ coefficients
        weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12))
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = y * np.sqrt(weights)
        updated = (
            np.linalg.pinv(weighted_design.T @ weighted_design + penalty)
            @ weighted_design.T
            @ weighted_target
        )
        if float(np.max(np.abs(updated - coefficients))) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def predict_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def normalize_features(
    train_x: np.ndarray, all_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (train_x - mean) / scale, (all_x - mean) / scale, np.vstack([mean, scale])


def grouped_cv_alpha(
    x: np.ndarray,
    y_norm: np.ndarray,
    chart_ids: list[str],
    seed: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    unique_charts = sorted(set(chart_ids), key=lambda chart_id: stable_score(chart_id, seed + 1))
    fold_count = min(5, len(unique_charts))
    if fold_count < 2:
        return 10.0, 0.01, [
            {
                "alpha": 10.0,
                "huber_delta": 0.01,
                "cv_mean_axis_normalized_error": math.nan,
                "fold_count": fold_count,
            }
        ]
    chart_fold = {chart_id: index % fold_count for index, chart_id in enumerate(unique_charts)}
    alpha_rows: list[dict[str, Any]] = []
    for delta in [0.0025, 0.005, 0.01, 0.02, 0.05]:
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            fold_errors: list[float] = []
            for fold in range(fold_count):
                validation = np.asarray([chart_fold[chart_id] == fold for chart_id in chart_ids])
                training = ~validation
                if not training.any() or not validation.any():
                    continue
                train_norm, validation_norm, _ = normalize_features(x[training], x[validation])
                coefficients = fit_huber_ridge(train_norm, y_norm[training], alpha, delta)
                predicted_norm = predict_ridge(validation_norm, coefficients)
                fold_errors.extend(np.abs(predicted_norm - y_norm[validation]).tolist())
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "huber_delta": delta,
                    "cv_mean_axis_normalized_error": float(np.mean(fold_errors))
                    if fold_errors
                    else math.nan,
                    "fold_count": fold_count,
                    "validation_observations": len(fold_errors),
                }
            )
    finite = [
        row for row in alpha_rows if math.isfinite(float(row["cv_mean_axis_normalized_error"]))
    ]
    best = (
        min(
            finite,
            key=lambda row: (
                float(row["cv_mean_axis_normalized_error"]),
                -float(row["alpha"]),
                float(row["huber_delta"]),
            ),
        )
        if finite
        else alpha_rows[0]
    )
    return float(best["alpha"]), float(best["huber_delta"]), alpha_rows


def metric_row(values: list[dict[str, Any]], method: str, split: str, chart_type: str = "all") -> dict[str, Any]:
    absolute = np.asarray([float(row[f"{method}_absolute_error"]) for row in values], dtype=float)
    relative = np.asarray([float(row[f"{method}_relative_error"]) for row in values], dtype=float)
    axis_normalized = np.asarray(
        [float(row[f"{method}_axis_normalized_error"]) for row in values], dtype=float
    )
    errors_signed = np.asarray(
        [float(row[f"{method}_value"]) - float(row["pseudo_gold"]) for row in values], dtype=float
    )
    return {
        "split": split,
        "chart_type": chart_type,
        "method": method,
        "n": len(values),
        "chart_count": len({row["chart_id"] for row in values}),
        "mae": float(absolute.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "rmse": float(np.sqrt(np.mean(errors_signed**2))),
        "mean_relative_error": float(relative.mean()),
        "median_relative_error": float(np.median(relative)),
        "mean_axis_normalized_error": float(axis_normalized.mean()),
        "median_axis_normalized_error": float(np.median(axis_normalized)),
        "within_1pct_axis": float(np.mean(axis_normalized <= 0.01)),
        "within_2pct_axis": float(np.mean(axis_normalized <= 0.02)),
        "within_5pct_axis": float(np.mean(axis_normalized <= 0.05)),
        "mean_signed_error": float(errors_signed.mean()),
    }


def bootstrap_mae_delta(rows: list[dict[str, Any]], seed: int, iterations: int = 5000) -> dict[str, Any]:
    by_chart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chart[str(row["chart_id"])].append(row)
    chart_ids = sorted(by_chart)
    raw = np.asarray(
        [np.mean([float(row["raw_absolute_error"]) for row in by_chart[chart_id]]) for chart_id in chart_ids]
    )
    calibrated = np.asarray(
        [
            np.mean([float(row["calibrated_absolute_error"]) for row in by_chart[chart_id]])
            for chart_id in chart_ids
        ]
    )
    delta = raw - calibrated
    raw_axis = np.asarray(
        [
            np.mean([float(row["raw_axis_normalized_error"]) for row in by_chart[chart_id]])
            for chart_id in chart_ids
        ]
    )
    calibrated_axis = np.asarray(
        [
            np.mean(
                [float(row["calibrated_axis_normalized_error"]) for row in by_chart[chart_id]]
            )
            for chart_id in chart_ids
        ]
    )
    delta_axis = raw_axis - calibrated_axis
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=float)
    bootstrap_axis = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, len(delta), size=len(delta))
        bootstrap[index] = float(delta[sampled].mean())
        bootstrap_axis[index] = float(delta_axis[sampled].mean())
    return {
        "chart_count": len(chart_ids),
        "raw_minus_calibrated_mae": float(delta.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "improvement_probability": float(np.mean(bootstrap > 0)),
        "raw_minus_calibrated_axis_normalized_mae": float(delta_axis.mean()),
        "axis_ci95_low": float(np.quantile(bootstrap_axis, 0.025)),
        "axis_ci95_high": float(np.quantile(bootstrap_axis, 0.975)),
        "axis_improvement_probability": float(np.mean(bootstrap_axis > 0)),
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_jsonl(args.samples)
    if len(rows) < 8:
        raise RuntimeError(f"Need at least 8 samples, found {len(rows)}")
    split_by_chart = assign_chart_splits(rows, args.seed, args.test_fraction)
    train_charts = {chart_id for chart_id, split in split_by_chart.items() if split == "train"}
    test_charts = {chart_id for chart_id, split in split_by_chart.items() if split == "test"}
    assert train_charts.isdisjoint(test_charts)

    x_all = np.vstack([feature_vector(row) for row in rows])
    gold = np.asarray([float(row["pseudo_gold"]) for row in rows])
    raw = np.asarray([float(row["raw_geometry_value"]) for row in rows])
    spans = np.asarray([max(float(row["axis"]["axis_span"]), 1e-12) for row in rows])
    y_normalized = (gold - raw) / spans
    train_mask = np.asarray([split_by_chart[str(row["chart_id"])] == "train" for row in rows])
    train_x_normalized, all_x_normalized, normalization = normalize_features(x_all[train_mask], x_all)
    best_alpha, best_huber_delta, cv_rows = grouped_cv_alpha(
        x_all[train_mask],
        y_normalized[train_mask],
        [str(row["chart_id"]) for row in np.asarray(rows, dtype=object)[train_mask]],
        args.seed,
    )
    coefficients = fit_huber_ridge(
        train_x_normalized,
        y_normalized[train_mask],
        best_alpha,
        best_huber_delta,
    )
    predicted_normalized = predict_ridge(all_x_normalized, coefficients)
    predicted_error = predicted_normalized * spans
    corrected = raw + predicted_error

    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        pseudo_gold = gold[index]
        axis_span = spans[index]
        raw_absolute = abs(raw[index] - pseudo_gold)
        calibrated_absolute = abs(corrected[index] - pseudo_gold)
        prediction = {
            "sample_id": row["sample_id"],
            "chart_id": row["chart_id"],
            "representative_sample_id": row["representative_sample_id"],
            "chart_type": row["geometry"]["chart_type"],
            "split": split_by_chart[str(row["chart_id"])],
            "pseudo_gold": pseudo_gold,
            "axis_span": axis_span,
            "raw_value": raw[index],
            "predicted_error": predicted_error[index],
            "calibrated_value": corrected[index],
            "raw_absolute_error": raw_absolute,
            "calibrated_absolute_error": calibrated_absolute,
            "raw_relative_error": raw_absolute / max(abs(pseudo_gold), 1e-12),
            "calibrated_relative_error": calibrated_absolute / max(abs(pseudo_gold), 1e-12),
            "raw_axis_normalized_error": raw_absolute / axis_span,
            "calibrated_axis_normalized_error": calibrated_absolute / axis_span,
            "raw_value_for_metric": raw[index],
            "calibrated_value_for_metric": corrected[index],
        }
        # Metric helper expects method_value names.
        prediction["raw_value"] = raw[index]
        prediction["calibrated_value"] = corrected[index]
        prediction_rows.append(prediction)
        source = cv2.imread(row["source_image_path"], cv2.IMREAD_COLOR)
        if source is not None:
            overlay = draw_debug_overlay(
                source,
                token_from_dict(row["label"]),
                row["axis"],
                row["geometry"],
                raw[index],
                pseudo_gold,
                corrected[index],
            )
            cv2.imwrite(row["debug_overlay_path"], overlay)

    overall_metrics: list[dict[str, Any]] = []
    type_metrics: list[dict[str, Any]] = []
    for split in ("train", "test", "all"):
        selected = prediction_rows if split == "all" else [row for row in prediction_rows if row["split"] == split]
        if not selected:
            continue
        for method in ("raw", "calibrated"):
            overall_metrics.append(metric_row(selected, method, split))
        for chart_type in sorted({row["chart_type"] for row in selected}):
            subset = [row for row in selected if row["chart_type"] == chart_type]
            for method in ("raw", "calibrated"):
                type_metrics.append(metric_row(subset, method, split, chart_type))

    test_rows = [row for row in prediction_rows if row["split"] == "test"]
    bootstrap = bootstrap_mae_delta(test_rows, args.seed + 17)
    split_rows = [
        {
            "chart_id": chart_id,
            "split": split,
            "sample_count": sum(str(row["chart_id"]) == chart_id for row in rows),
            "chart_types": "|".join(
                sorted({row["geometry"]["chart_type"] for row in rows if str(row["chart_id"]) == chart_id})
            ),
        }
        for chart_id, split in sorted(split_by_chart.items())
    ]
    write_csv(args.project / "results" / "calibrated_predictions.csv", prediction_rows)
    write_csv(args.project / "results" / "metrics_overall.csv", overall_metrics)
    write_csv(args.project / "results" / "metrics_by_type.csv", type_metrics)
    write_csv(args.project / "results" / "calibration_cv.csv", cv_rows)
    write_csv(args.project / "results" / "chart_split.csv", split_rows)
    write_csv(args.project / "results" / "calibration_bootstrap.csv", [bootstrap])
    model = {
        "model": "huber_ridge_error_calibrator",
        "target": "(pseudo_gold - raw_geometry_value) / axis_span",
        "output": "predicted_error = predicted_normalized_error * axis_span",
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "best_alpha": best_alpha,
        "best_huber_delta": best_huber_delta,
        "feature_names": FEATURE_NAMES,
        "feature_mean": normalization[0].tolist(),
        "feature_scale": normalization[1].tolist(),
        "coefficients": coefficients.tolist(),
        "train_charts": len(train_charts),
        "test_charts": len(test_charts),
        "train_test_chart_overlap": len(train_charts & test_charts),
        "bootstrap_test_mae_delta": bootstrap,
    }
    (args.project / "audit" / "calibrator_model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(model, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
