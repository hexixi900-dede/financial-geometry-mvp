from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from phase3_line_core import ground_question_to_x


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_phase1_frozen(project: Path) -> int:
    checked = 0
    for line in (project / "audit" / "frozen_results.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = project / relative
        assert path.exists(), f"missing frozen Phase1 file: {relative}"
        assert sha256(path) == expected, f"Phase1 hash changed: {relative}"
        checked += 1
    return checked


def assert_auto_signature(project: Path) -> None:
    source = (project / "src" / "phase3_line_core.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "auto_geometry_pipeline"]
    assert len(functions) == 1
    arguments = [argument.arg for argument in functions[0].args.args]
    assert arguments == ["reader", "masked_bgr", "question"], arguments
    function_source = ast.get_source_segment(source, functions[0]) or ""
    forbidden = ["pseudo_gold", "masked_value_label", "label.center", "label.cx"]
    assert not any(term in function_source for term in forbidden), "auto parser contains a forbidden construction-only input"


def assert_grounding_fallbacks() -> None:
    anchors = [
        {
            "label": "FY23",
            "canonical": "FY23",
            "normalized": "fy23",
            "center_x": 100.0,
            "confidence": 0.99,
            "scalar": 2023.0,
            "scalar_family": "year",
        },
        {
            "label": "FY25",
            "canonical": "FY25",
            "normalized": "fy25",
            "center_x": 200.0,
            "confidence": 0.99,
            "scalar": 2025.0,
            "scalar_family": "year",
        },
    ]
    direct = ground_question_to_x("What is the value for FY23?", anchors)
    assert direct["status"] == "ok" and direct["mode"] == "direct_anchor" and direct["predicted_target_x"] == 100.0
    interpolated = ground_question_to_x("What is the value for FY24?", anchors)
    assert interpolated["status"] == "ok" and interpolated["mode"] == "piecewise_linear_interpolation"
    assert abs(float(interpolated["predicted_target_x"]) - 150.0) < 1e-9
    outside = ground_question_to_x("What is the value for FY26?", anchors)
    assert outside["status"] == "x_grounding_unrecoverable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()

    phase1_checked = assert_phase1_frozen(project)
    phase2 = subprocess.run(
        [sys.executable, str(project / "src" / "validate_phase2.py"), "--project", str(project)],
        check=True,
        capture_output=True,
        text=True,
    )
    phase2_status = json.loads(phase2.stdout.strip().splitlines()[-1])
    assert phase2_status["status"] == "ok"
    phase2_protected = [
        "PHASE2_REPORT.md",
        "config/phase2_manual_review.json",
        "config/phase2_protocol.json",
        "examples/phase2_debug_overlays",
        "results/phase2_chart_split.csv",
        "results/phase2_failure_audit.csv",
        "results/phase2_manual_audit.csv",
        "results/phase2_metrics.csv",
        "results/phase2_paired_comparisons.csv",
        "results/phase2_sample_audit.csv",
        "results/phase2_screening.csv",
        "scripts/run_phase2_cpu.sh",
        "src/evaluate_phase2.py",
        "src/phase2_natural_bar.py",
        "src/render_phase2_overlays.py",
        "src/validate_phase2.py",
        "tests/test_phase2.py",
    ]
    git_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *phase2_protected], cwd=project)
    assert git_diff.returncode == 0, "a protected Phase2 result/code file was modified"

    assert_auto_signature(project)
    assert_grounding_fallbacks()
    samples = read_jsonl(project / "phase3_audit" / "line_samples.jsonl")
    assert samples, "no Phase3 samples"
    assert len({row["chart_id"] for row in samples}) >= 2
    for row in samples:
        contract = row["parser_runtime_contract"]
        assert contract["inputs"] == ["masked_chart", "question"]
        assert "pseudo_gold" in contract["forbidden_inputs"]
        assert row["mask_overlaps_protected_series_pixels"] == 0
        assert row["mask_pixel_count"] > 0
        assert row["axis"] is not None
        assert len(row["axis"]["ticks"]) >= 3
        assert row["x_axis_anchors"]
        assert row["grounding"]["status"] == "ok"
        assert row["grounding"]["mode"] in {"direct_anchor", "piecewise_linear_interpolation"}
        assert row["oracle_x_geometry"] is not None
        assert row["auto_x_local_geometry"] is not None
        assert row["auto_x_local_geometry"]["fitted_segments"]
        assert math.isfinite(float(row["auto_x_local_value"]))

    flat = read_csv(project / "results" / "phase3_line_samples.csv")
    assert len(flat) == len(samples)
    assert all(int(row["parser_used_hidden_bbox_or_gold"]) == 0 for row in flat)
    split_rows = read_csv(project / "results" / "phase3_chart_split.csv")
    train = {row["chart_id"] for row in split_rows if row["split"] == "train"}
    test = {row["chart_id"] for row in split_rows if row["split"] == "test"}
    assert train and test and not (train & test)

    metrics = {row["method"]: row for row in read_csv(project / "results" / "phase3_method_metrics.csv")}
    method_fields = {
        "oracle_x_old_phase1_local_fit": "oracle_x_value",
        "auto_x_column": "auto_x_column_value",
        "auto_x_local_line_fit": "auto_x_local_value",
    }
    for method, field in method_fields.items():
        eligible = [row for row in flat if row[field] != ""]
        recomputed = sum(abs(float(row[field]) - float(row["pseudo_gold"])) for row in eligible) / len(eligible)
        assert abs(recomputed - float(metrics[method]["mae"])) < 1e-10

    overlays = list((project / "phase3_debug_overlays").glob("*.png"))
    assert len(overlays) >= 30
    assert all(path.stat().st_size > 0 for path in overlays)
    for row in samples:
        for suffix in ["oracle_x", "auto_column", "auto_local"]:
            assert (project / "phase3_debug_overlays" / f"{row['sample_id']}_{suffix}.png").exists()
    visual = read_csv(project / "results" / "phase3_visual_audit.csv")
    assert {row["sample_id"] for row in visual} == {row["sample_id"] for row in samples}
    assert all(row["target_grounding_correct"] == "1" for row in visual)
    assert all(row["series_match_correct"] == "1" for row in visual)

    required = [
        "results/phase3_chart_screening.csv",
        "results/phase3_line_samples.csv",
        "results/phase3_method_metrics.csv",
        "results/phase3_pairwise_metrics.csv",
        "results/phase3_local_fitting_effects.csv",
        "results/phase3_calibration_bias_metrics.csv",
        "results/phase3_failure_attribution.csv",
        "results/phase3_summary.json",
        "results/phase3_reproducibility.json",
        "PHASE3_LINE_2D_REPORT.md",
    ]
    assert all((project / relative).exists() for relative in required)
    reproducibility = json.loads((project / "results" / "phase3_reproducibility.json").read_text(encoding="utf-8"))
    assert reproducibility["verdict"] == "REPRODUCIBLE" and reproducibility["exact_hash_match"] is True
    published_overlays = list((project / "examples" / "phase3_debug_overlays").glob("*.png"))
    assert 30 <= len(published_overlays) <= 50
    source_text = "\n".join((project / "src" / name).read_text(encoding="utf-8") for name in ["phase3_line_core.py", "run_phase3_line_2d.py"])
    assert "/data/" not in source_text and "/home/" not in source_text

    summary = json.loads((project / "results" / "phase3_summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == len(samples)
    assert summary["chart_count"] == len({row["chart_id"] for row in samples})
    assert summary["overlay_count"] == len(overlays)
    assert summary["train_test_chart_overlap"] == 0
    print(
        json.dumps(
            {
                "status": "ok",
                "phase1_frozen_files_checked": phase1_checked,
                "phase2_status": phase2_status["status"],
                "phase3_samples": len(samples),
                "phase3_charts": len({row["chart_id"] for row in samples}),
                "direct_grounding": summary["direct_grounding_samples"],
                "interpolated_grounding": summary["interpolated_grounding_samples"],
                "overlays": len(overlays),
                "chart_split_overlap": 0,
                "parser_hidden_input_violations": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
