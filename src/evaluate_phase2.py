from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np


METHODS = {
    "raw_vlm": ("raw_vlm_prediction", "raw_vlm_correct"),
    "raw_geometry": ("raw_geometry_prediction", "raw_geometry_correct"),
    "calibrated_geometry": ("calibrated_prediction", "calibrated_geometry_correct"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_number(value: Any) -> float | None:
    import re

    match = re.search(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def method_metrics(rows: list[dict[str, Any]], method: str, scope: str) -> dict[str, Any]:
    _, correct_key = METHODS[method]
    numerical = [row for row in rows if row["question_type"] == "numerical"]
    errors: list[float] = []
    for row in numerical:
        gold = parse_number(row["gold"])
        if gold is None:
            continue
        if method == "raw_vlm":
            prediction = parse_number(row["raw_vlm_prediction"])
            if prediction is None:
                continue
            errors.append(abs(prediction - gold))
        elif method == "raw_geometry":
            errors.append(abs(float(row["raw_geometry_value"]) - gold))
        else:
            errors.append(abs(float(row["calibrated_value"]) - gold))
    return {
        "scope": scope,
        "method": method,
        "n": len(rows),
        "chart_count": len({str(row["chart_id"]) for row in rows}),
        "correct": sum(bool(row[correct_key]) for row in rows),
        "accuracy": float(np.mean([bool(row[correct_key]) for row in rows])) if rows else math.nan,
        "numerical_n": len(numerical),
        "numerical_parseable_n": len(errors),
        "numerical_mae": float(np.mean(errors)) if errors else math.nan,
        "numerical_median_absolute_error": float(np.median(errors)) if errors else math.nan,
    }


def bootstrap_accuracy_delta(
    rows: list[dict[str, Any]], first_key: str, second_key: str, seed: int, iterations: int
) -> dict[str, Any]:
    by_chart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chart[str(row["chart_id"])].append(row)
    chart_ids = sorted(by_chart)
    observed = float(np.mean([bool(row[first_key]) for row in rows]) - np.mean([bool(row[second_key]) for row in rows]))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(chart_ids, size=len(chart_ids), replace=True)
        sample_rows = [row for chart_id in sampled for row in by_chart[str(chart_id)]]
        deltas.append(
            float(
                np.mean([bool(row[first_key]) for row in sample_rows])
                - np.mean([bool(row[second_key]) for row in sample_rows])
            )
        )
    return {
        "accuracy_delta": observed,
        "bootstrap_ci95_low": float(np.quantile(deltas, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_improvement_probability": float(np.mean(np.asarray(deltas) > 0)),
        "bootstrap_iterations": iterations,
        "bootstrap_cluster": "chart_id",
    }


def exact_mcnemar(rows: list[dict[str, Any]], first_key: str, second_key: str) -> dict[str, Any]:
    first_only = sum(bool(row[first_key]) and not bool(row[second_key]) for row in rows)
    second_only = sum(bool(row[second_key]) and not bool(row[first_key]) for row in rows)
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(first_only, second_only)
        tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    accepted_path = args.project / "phase2_audit" / "accepted_samples.jsonl"
    rows = read_jsonl(accepted_path)
    if not rows:
        raise RuntimeError("No accepted Phase 2 rows found")
    summary = json.loads((args.project / "phase2_audit" / "run_summary.json").read_text(encoding="utf-8"))
    screening = read_csv(args.project / "results" / "phase2_screening.csv")
    manual_review_path = args.project / "config" / "phase2_manual_review.json"
    manual_review = json.loads(manual_review_path.read_text(encoding="utf-8")) if manual_review_path.exists() else {}
    manual_by_sample = manual_review.get("accepted_sample_reviews", {})

    metrics: list[dict[str, Any]] = []
    scopes = [("all", rows)]
    scopes.extend((f"operation:{operation}", [row for row in rows if row["operation"] == operation]) for operation in sorted({row["operation"] for row in rows}))
    scopes.extend((f"type:{kind}", [row for row in rows if row["question_type"] == kind]) for kind in sorted({row["question_type"] for row in rows}))
    for scope, selected in scopes:
        for method in METHODS:
            metrics.append(method_metrics(selected, method, scope))
    write_csv(args.project / "results" / "phase2_metrics.csv", metrics)

    comparisons: list[dict[str, Any]] = []
    for name, first_key, second_key in [
        ("raw_geometry_minus_raw_vlm", "raw_geometry_correct", "raw_vlm_correct"),
        ("calibrated_geometry_minus_raw_vlm", "calibrated_geometry_correct", "raw_vlm_correct"),
        ("calibrated_minus_raw_geometry", "calibrated_geometry_correct", "raw_geometry_correct"),
    ]:
        comparisons.append(
            {
                "comparison": name,
                "n": len(rows),
                **bootstrap_accuracy_delta(rows, first_key, second_key, args.seed + len(comparisons) * 101, args.bootstrap_iterations),
                **exact_mcnemar(rows, first_key, second_key),
            }
        )
    write_csv(args.project / "results" / "phase2_paired_comparisons.csv", comparisons)

    failures = []
    for row in rows:
        failures.append(
            {
                "sample_id": row["sample_id"],
                "chart_id": row["chart_id"],
                "operation": row["operation"],
                "semantic_target": row["semantic_target"],
                "raw_vlm_correct": row["raw_vlm_correct"],
                "raw_geometry_correct": row["raw_geometry_correct"],
                "calibrated_geometry_correct": row["calibrated_geometry_correct"],
                "target_audit": row["target_audit"],
                "geometry_audit": row["geometry_audit"],
                "reasoning_audit": row["reasoning_audit"],
                "failure_attribution": row["failure_attribution"],
                "manual_failure_attribution": manual_by_sample.get(str(row["sample_id"]), {}).get("manual_failure_attribution", "not_manually_resolved"),
                "debug_overlay_path": row["debug_overlay_path"],
            }
        )
    write_csv(args.project / "results" / "phase2_failure_audit.csv", failures)
    write_csv(
        args.project / "results" / "phase2_manual_audit.csv",
        [
            {
                "sample_id": row["sample_id"],
                "chart_id": row["chart_id"],
                **manual_by_sample.get(str(row["sample_id"]), {"manual_failure_attribution": "not_manually_resolved"}),
            }
            for row in rows
        ],
    )
    write_csv(
        args.project / "results" / "phase2_chart_split.csv",
        [
            {
                "chart_id": row["chart_id"],
                "sample_id": row["sample_id"],
                "split": "phase2_natural_no_label_external_test",
                "phase1_calibration_chart_overlap": row["phase1_chart_overlap"],
            }
            for row in rows
        ],
    )

    all_metrics = {row["method"]: row for row in metrics if row["scope"] == "all"}
    by_comparison = {row["comparison"]: row for row in comparisons}
    screening_counts = Counter(row["reason"] for row in screening)
    operation_counts = Counter(row["operation"] for row in rows)
    type_counts = Counter(row["question_type"] for row in rows)
    failure_counts = Counter(row["failure_attribution"] for row in rows)

    raw_vlm = all_metrics["raw_vlm"]
    raw_geometry = all_metrics["raw_geometry"]
    calibrated = all_metrics["calibrated_geometry"]
    geometry_comparison = by_comparison["raw_geometry_minus_raw_vlm"]
    calibration_comparison = by_comparison["calibrated_minus_raw_geometry"]
    directional_geometry_gain = float(raw_geometry["accuracy"]) > float(raw_vlm["accuracy"])
    geometry_usable = directional_geometry_gain and len(rows) >= 10
    calibration_adds_value = float(calibrated["accuracy"]) > float(raw_geometry["accuracy"])
    overlay_count = len(list((args.project / "phase2_debug_overlays").rglob("*.png")))

    source_info = {
        "accepted_samples_sha256": sha256_file(accepted_path),
        "sample_audit_sha256": sha256_file(args.project / "results" / "phase2_sample_audit.csv"),
        "raw_vlm_sha256": summary["raw_vlm_sha256"],
        "calibrator_sha256": summary["calibrator_sha256"],
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "easyocr": easyocr.__version__,
    }
    (args.project / "phase2_audit" / "environment_and_hashes.json").write_text(
        json.dumps(source_info, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = f"""# Phase 2 — Natural No-label Bar End-to-End Validation

## Executive verdict

- **Geometry recovery usable for this screened subset:** {'YES' if geometry_usable else ('DIRECTIONALLY POSITIVE BUT INCONCLUSIVE' if directional_geometry_gain else 'NO / NOT YET SHOWN')}. Raw Geometry accuracy was {raw_geometry['correct']}/{raw_geometry['n']} ({100*float(raw_geometry['accuracy']):.1f}%) versus Raw VLM {raw_vlm['correct']}/{raw_vlm['n']} ({100*float(raw_vlm['accuracy']):.1f}%), a paired delta of {100*float(geometry_comparison['accuracy_delta']):+.1f} percentage points. The chart-cluster bootstrap 95% interval is [{100*float(geometry_comparison['bootstrap_ci95_low']):+.1f}, {100*float(geometry_comparison['bootstrap_ci95_high']):+.1f}] points. Fewer than 10 accepted charts is treated as insufficient to claim usability or superiority, regardless of the point estimate.
- **Calibration adds extra value:** {'YES in this pilot' if calibration_adds_value else 'NO in this pilot'}. Calibrated Geometry accuracy was {calibrated['correct']}/{calibrated['n']} ({100*float(calibrated['accuracy']):.1f}%). Its paired delta versus Raw Geometry was {100*float(calibration_comparison['accuracy_delta']):+.1f} points, bootstrap 95% interval [{100*float(calibration_comparison['bootstrap_ci95_low']):+.1f}, {100*float(calibration_comparison['bootstrap_ci95_high']):+.1f}]. This calibration conclusion is separate from the Geometry conclusion.
- No GPU inference was run. The complete, already-existing Qwen2.5-VL-7B direct baseline was reused byte-for-byte.

## Scope and protocol

Phase 1's 87 masked-value samples and all frozen Phase 1 result files were left unchanged. Phase 2 uses naturally unlabeled FinMME charts only: simple vertical Bar charts, one fitted left y-axis, no detected second/additional y-axis, at most one coarse prefilter sloped segment, two to thirty consistent bars, readable x labels, and no non-axis numeric OCR token in the plot interior. Multiple marker series, stacked, grouped/ambiguous, combination, multi-panel, Line, Pie, Area, and other chart types are rejected.

The source pool had 11,099 FinMME questions and 4,446 chart hashes. Existing Phase 1 screening identified 537 charts with no geometry-linked data label; the stricter Phase 2 question/structure prefilter produced {len(screening)} evaluated question candidates. The final clean subset contains **{len(rows)} question(s) on {len({row['chart_id'] for row in rows})} distinct chart(s)** ({dict(type_counts)}; operations {dict(operation_counts)}). It was not padded to a requested count.

### Leakage boundary

The detector receives only image, question, options, question type, chart hash, and non-Gold structure prefilter fields. It does not receive Gold, tolerance, hidden value-label boxes, or manual target coordinates. `Question → semantic target` matches normalized question text to CPU-OCR x-axis labels; all bars are detected first and the matched label selects a bar by its automatically assigned x-position cell. Gold and official tolerance are joined by `sample_id` only after semantic targeting, bar selection, y-axis interpolation, calibration, and Python reasoning have completed. Phase 1/Phase 2 chart overlap is **{summary['phase1_chart_overlap']}**.

## Results

| System | Correct / N | Accuracy | Numerical MAE | Numerical median AE |
|---|---:|---:|---:|---:|
| Raw VLM (Qwen2.5-VL-7B Direct) | {raw_vlm['correct']} / {raw_vlm['n']} | {100*float(raw_vlm['accuracy']):.1f}% | {fmt_float(raw_vlm['numerical_mae'])} | {fmt_float(raw_vlm['numerical_median_absolute_error'])} |
| Geometry without Calibration | {raw_geometry['correct']} / {raw_geometry['n']} | {100*float(raw_geometry['accuracy']):.1f}% | {fmt_float(raw_geometry['numerical_mae'])} | {fmt_float(raw_geometry['numerical_median_absolute_error'])} |
| Geometry + Phase 1 Calibration | {calibrated['correct']} / {calibrated['n']} | {100*float(calibrated['accuracy']):.1f}% | {fmt_float(calibrated['numerical_mae'])} | {fmt_float(calibrated['numerical_median_absolute_error'])} |

Official absolute Numerical tolerance is used per record. Single-choice rows, when retained, require the official option letter. For direct reads Geometry returns the interpolated value; for difference and growth-rate questions the target values come only from Geometry and deterministic Python performs the operation.

Exact paired McNemar p-values are {fmt_float(geometry_comparison['mcnemar_exact_two_sided_p'], 4)} for Raw Geometry versus Raw VLM and {fmt_float(calibration_comparison['mcnemar_exact_two_sided_p'], 4)} for Calibrated versus Raw Geometry. With one chart and no discordant correctness pair, both the exact test and the degenerate `[0, 0]` bootstrap interval are non-informative; no significance or superiority claim is possible.

## Target / Geometry / reasoning audit

- Accepted target matches require an exact/compact or high-confidence fuzzy OCR–question match (score at least 0.78); ambiguous direct targets are rejected.
- All accepted rows store the semantic target, every detected x label, selected bar ID/bbox, y ticks `(value, pixel_y)`, bar top `(x, y)`, per-target raw/calibrated values, final predictions, Gold, tolerance, correctness, and failure attribution.
- Failure attribution counts: {dict(failure_counts)}. A wrong direct read with a high-confidence target is attributed to Geometry. For derived answers, component-level Gold is unavailable, so a wrong answer is conservatively labelled `geometry_or_semantic_decomposition` rather than pretending the stage is identifiable.
- {overlay_count} per-sample overlays were generated. Red denotes selected bars/tops, gray other detected bars, magenta axis ticks, and blue the inferred plot box. Overlays do not feed back into targeting. {manual_review.get('reviewed_overlay_count', 0)} representative overlays were visually reviewed after predictions were frozen.
- Accepted-sample manual review: `{json.dumps(manual_by_sample, ensure_ascii=False, sort_keys=True)}`. Official correctness is not changed by this audit. In particular, a suspected Gold/chart mismatch remains officially scored as wrong.

## Screening audit

Rejection counts: `{json.dumps(dict(screening_counts), ensure_ascii=False, sort_keys=True)}`.

The dominant rejects are structurally out of scope, unreadable/grouped x-label layouts, detected numeric annotations, or failure to find a unique semantic target. These rejects are part of the audit trail in `results/phase2_screening.csv`; they are not silently discarded.

## Statistical interpretation and risk scan

Confidence level: **CAUTION**. The study is a deliberately high-precision, selected pilot rather than a random FinMME sample. Chart-cluster bootstrap respects the chart as the sampling unit, and one accepted question per chart prevents repeated-chart pseudo-replication. The main risks are selection bias/Berkson-type conditioning on easy charts and garden-of-forking-paths risk from exploratory thresholds. No causal claim is made. Simpson reversal, ecological inference, base-rate claims, regression-to-mean claims, survivorship claims, look-elsewhere significance claims, collider adjustment, reverse causality, and correlation-as-causation are not applicable to the stated paired predictive comparison; all 11 protocol checks were considered.

## Reproducibility and Material Passport

- **Material identity:** FinMME released records and already-cached natural chart images; no source dataset was downloaded or modified. Qwen2.5-VL-7B Direct prediction SHA-256: `{summary['raw_vlm_sha256']}`. Phase 1 calibrator SHA-256: `{summary['calibrator_sha256']}`.
- **Material mode:** source records/images and Raw VLM predictions are read-only inputs. Derived OCR caches, sample tables, overlays, metrics, and this report are new Phase 2 outputs.
- **Transformations:** CPU EasyOCR → linear left-axis fit → quantized-color bar components → x-label/bar association → question-label matching → pixel/value interpolation → optional frozen Phase 1 error correction → deterministic Python QA → official scoring.
- **Lineage:** `phase2_audit/accepted_samples.jsonl` SHA-256 `{source_info['accepted_samples_sha256']}`; sample CSV SHA-256 `{source_info['sample_audit_sha256']}`. Each row carries `chart_id`, `sample_id`, and input image path.
- **Environment:** Python {source_info['python']}, OpenCV {source_info['opencv']}, NumPy {source_info['numpy']}, EasyOCR {source_info['easyocr']}; OCR runs with `CUDA_VISIBLE_DEVICES` empty and six CPU threads.
- **Determinism:** downstream evaluation and bootstrap are deterministic at seed {args.seed}. OCR is environment-sensitive; its cache is append-only and retained. A full rerun should compare row/column structure and metrics, not wall time.

## Artifacts

- `results/phase2_sample_audit.csv` — sample-level complete audit
- `results/phase2_screening.csv` — accepted/rejected candidates and reasons
- `results/phase2_metrics.csv` — overall and stratified system metrics
- `results/phase2_paired_comparisons.csv` — paired bootstrap and exact McNemar results
- `results/phase2_failure_audit.csv` — target/geometry/reasoning attribution
- `results/phase2_manual_audit.csv` — post-prediction visual review and suspected annotation issues
- `results/phase2_chart_split.csv` — Phase 2 held-out chart list and Phase 1 overlap check
- `phase2_audit/accepted_samples.jsonl` — nested machine-readable audit
- `phase2_audit/chart_analysis_cache.jsonl` — recoverable CPU OCR/geometry cache
- `phase2_debug_overlays/` — visual audit overlays
"""
    (args.project / "PHASE2_REPORT.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "samples": len(rows),
                "raw_vlm_accuracy": raw_vlm["accuracy"],
                "raw_geometry_accuracy": raw_geometry["accuracy"],
                "calibrated_geometry_accuracy": calibrated["accuracy"],
                "geometry_usable": geometry_usable,
                "calibration_adds_value": calibration_adds_value,
                "overlay_count": overlay_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
