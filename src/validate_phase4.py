"""Phase 4 validator: frozen prior phases, no-leak contract, metrics, overlays.

Runs the Phase 3 validator first (which itself checks the 12 frozen Phase 1
hashes and the Phase 2 validator), then verifies Phase 4-specific contracts:
parser signature/no-leak AST, sample contracts, chart-disjoint split, metric
recomputation, overlay coverage for accepted AND rejected samples, and the
deterministic rerun verdict.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_auto_signature(project: Path) -> None:
    source = (project / "src" / "phase4_multiline_core.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "auto_multiline_pipeline"]
    assert len(functions) == 1
    arguments = [argument.arg for argument in functions[0].args.args]
    assert arguments == ["reader", "masked_bgr", "question", "oracle_series_name"], arguments
    defaults = functions[0].args.defaults
    assert len(defaults) == 1 and isinstance(defaults[0], ast.Constant) and defaults[0].value is None
    body = ast.get_source_segment(source, functions[0]) or ""
    forbidden = ["pseudo_gold", "masked_value_label", "label.center", "label.cx", "construction"]
    assert not any(term in body for term in forbidden), "auto pipeline references a forbidden construction-only input"

    runner = (project / "src" / "run_phase4_multiline.py").read_text(encoding="utf-8")
    runner_tree = ast.parse(runner)
    auto_calls = 0
    for node in ast.walk(runner_tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "auto_multiline_pipeline"):
            continue
        auto_calls += 1
        oracle_keywords = [keyword for keyword in node.keywords if keyword.arg == "oracle_series_name"]
        if not oracle_keywords:
            # Auto-Series configuration: exactly three positional arguments.
            assert len(node.args) == 3, "Auto-Series call must pass only reader, masked image, question"
    assert auto_calls >= 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()

    phase3 = subprocess.run(
        [sys.executable, str(project / "src" / "validate_phase3.py"), "--project", str(project)],
        check=True,
        capture_output=True,
        text=True,
    )
    phase3_status = json.loads(phase3.stdout.strip().splitlines()[-1])
    assert phase3_status["status"] == "ok"

    phase3_protected = [
        "PHASE3_LINE_2D_REPORT.md",
        "config/phase3_line_chart_decisions.csv",
        "results/phase3_bias_constants.csv",
        "results/phase3_calibration_bias_metrics.csv",
        "results/phase3_chart_screening.csv",
        "results/phase3_chart_split.csv",
        "results/phase3_failure_attribution.csv",
        "results/phase3_line_samples.csv",
        "results/phase3_local_fitting_effects.csv",
        "results/phase3_method_metrics.csv",
        "results/phase3_pairwise_metrics.csv",
        "results/phase3_reproducibility.json",
        "results/phase3_summary.json",
        "results/phase3_visual_audit.csv",
        "src/phase3_line_core.py",
        "src/run_phase3_line_2d.py",
        "src/validate_phase3.py",
        "src/verify_phase3_reproducibility.py",
    ]
    git_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *phase3_protected], cwd=project)
    assert git_diff.returncode == 0, "a protected Phase3 result/code file was modified"

    assert_auto_signature(project)

    samples = read_jsonl(project / "phase4_audit" / "line_samples.jsonl")
    assert samples, "no Phase4 samples"
    for row in samples:
        contract = row["parser_runtime_contract"]
        assert contract["inputs"] == ["masked_chart", "question"]
        assert "pseudo_gold" in contract["forbidden_inputs"]
        assert row["mask_overlaps_protected_series_pixels"] == 0
        assert row["mask_pixel_count"] > 0
        auto = row.get("auto_result") or {}
        assert row.get("auto_status") in {
            "success",
            "ambiguous_series",
            "x_grounding_unrecoverable",
            "line_localization_failed",
            "axis_mapping_failed",
        }
        if row.get("auto_status") == "success":
            assert auto.get("axis") and len(auto["axis"]["ticks"]) >= 3
            assert auto.get("x_axis_anchors")
            assert (auto.get("grounding") or {}).get("status") == "ok"
        if row["chart_kind"] == "multi_line":
            assert row.get("oracle_result") is not None, "multi-line samples must carry the Oracle-Series diagnostic"
            assert row.get("target_series_name"), "multi-line samples must name the target series"

    flat = read_csv(project / "results" / "phase4_line_samples.csv")
    assert len(flat) == len(samples)
    assert all(int(row["parser_used_hidden_bbox_or_gold"]) == 0 for row in flat)
    split_rows = read_csv(project / "results" / "phase4_chart_split.csv")
    train = {row["chart_id"] for row in split_rows if row["split"] == "train"}
    test = {row["chart_id"] for row in split_rows if row["split"] == "test"}
    assert train and test and not (train & test)

    metrics = {(row["subset"], row["method"]): row for row in read_csv(project / "results" / "phase4_method_metrics.csv")}
    fields = {
        "auto_column": "auto_column_value",
        "auto_local": "auto_local_value",
        "oracle_column": "oracle_column_value",
        "oracle_local": "oracle_local_value",
    }
    for (subset, method), metric in metrics.items():
        field = fields[method]
        if subset == "single_line":
            eligible = [row for row in flat if row["chart_kind"] == "single_line" and row[field] != ""]
        elif subset == "multi_line":
            eligible = [row for row in flat if row["chart_kind"] == "multi_line" and row[field] != ""]
        else:
            eligible = [row for row in flat if row[field] != ""]
        if not eligible:
            assert int(metric["eligible_samples"]) == 0
            continue
        recomputed = sum(abs(float(row[field]) - float(row["pseudo_gold"])) for row in eligible) / len(eligible)
        assert abs(recomputed - float(metric["mae"])) < 1e-10, (subset, method)

    overlay_dir = project / "phase4_debug_overlays"
    overlays = list(overlay_dir.glob("*.png"))
    assert all(path.stat().st_size > 0 for path in overlays)
    for row in samples:
        for suffix in ["auto_column", "auto_local"]:
            assert (overlay_dir / f"{row['sample_id']}_{suffix}.png").exists()
        if row["chart_kind"] == "multi_line":
            for suffix in ["oracle_column", "oracle_local"]:
                assert (overlay_dir / f"{row['sample_id']}_{suffix}.png").exists()

    rejection_audit = read_csv(project / "results" / "phase4_rejection_audit.csv")
    runtime_rejected = [row for row in rejection_audit if row["stage"] == "runtime"]
    expected_rejected = sum(row.get("auto_status") != "success" for row in samples)
    assert len(runtime_rejected) == expected_rejected

    reproducibility = json.loads((project / "results" / "phase4_reproducibility.json").read_text(encoding="utf-8"))
    assert reproducibility["verdict"] == "REPRODUCIBLE" and reproducibility["exact_hash_match"] is True

    summary = json.loads((project / "results" / "phase4_summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == len(samples)
    assert summary["train_test_chart_overlap"] == 0
    print(
        json.dumps(
            {
                "status": "ok",
                "phase3_validator": phase3_status["status"],
                "phase4_samples": len(samples),
                "phase4_charts": len({row["chart_id"] for row in samples}),
                "single_line_samples": summary["single_line_samples"],
                "multi_line_samples": summary["multi_line_samples"],
                "overlays": len(overlays),
                "runtime_rejected_samples": expected_rejected,
                "chart_split_overlap": 0,
                "parser_hidden_input_violations": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
