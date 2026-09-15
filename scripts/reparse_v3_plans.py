#!/usr/bin/env python3
"""Reuse stored v3 model replies; only normalize their JSON again on the CPU.

No image measurement, model generation, answer scoring, or Gold data is used.
The runner's feedback eligibility determines the predicted number entering the
enhanced workflow; this is separate from the accepted Geometry-plan count.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / "src"))
from vlm_semantic_planner import extract_json, normalize_plan
from run_visual_feedback import feedback_eligible

VERSION = "visual_feedback_v4"
NORMALIZER_FILE = RELEASE / "src/vlm_semantic_planner.py"

def records(path):
    return [json.loads(line) for line in path.open() if line.strip()]

def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data))
    temp.replace(path)

def route(record):
    plan = record.get("plan") or {}
    return dict(status=record["status"], error=record.get("error"),
                route_geometry=bool(plan.get("route_geometry")),
                route_reason=plan.get("route_reason"),
                feedback_eligible=feedback_eligible(record, {}),
                has_direct_value_labels=plan.get("has_direct_value_labels"),
                chart_type=plan.get("chart_type"), y_axis_count=plan.get("y_axis_count"),
                operation=plan.get("operation"), target_count=len(plan.get("targets") or []))

def counts(audits, which):
    states = [row[which] for row in audits]
    return dict(n=len(states), accepted_geometry_plans=sum(x["route_geometry"] for x in states),
                feedback_eligible=sum(x["feedback_eligible"] for x in states),
                raw_initial_route=sum(not x["feedback_eligible"] for x in states),
                statuses=dict(Counter(x["status"] for x in states)),
                route_reasons=dict(Counter(str(x["route_reason"]) for x in states)),
                parse_errors=dict(Counter(x["error"] for x in states if x["status"] == "plan_parse_failed")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.output.resolve()
    if source == destination:
        raise ValueError("source_and_output_must_differ")
    for cohort in ("pilot", "full"):
        for journal in ("measurements_initial", "feedback_initial", "measurements", "replies"):
            path = destination / cohort / (journal + ".jsonl")
            if path.exists() and path.stat().st_size:
                raise RuntimeError("Output already contains experiment work: " + str(path))
    original = records(source / "full/plans.jsonl")
    sources = {str(row["sample_id"]): row for row in records(source / "full/geometry_inputs.jsonl")}
    if len(original) != 3699 or len(sources) != 3699 or len({str(x["sample_id"]) for x in original}) != 3699:
        raise ValueError("expected_3699_unique_plans_and_sources")
    if {str(x["sample_id"]) for x in original} != set(sources):
        raise ValueError("source_plan_id_mismatch")
    normalized, audits = [], []
    semantic_keys = {"chart_type", "y_axis_count", "has_direct_value_labels", "operation",
                     "targets", "reason", "calculations"}
    for old in original:
        sid = str(old["sample_id"])
        plan, error, status, nested_keys = {}, None, "success", []
        if old.get("status") == "inference_error":
            status, error = "inference_error", old.get("error")
        else:
            try:
                raw_plan = extract_json(old["raw_text"])
                context = raw_plan.get("chart_context")
                if isinstance(context, dict):
                    nested_keys = sorted(semantic_keys & set(context))
                plan = normalize_plan(raw_plan, sources[sid]["question"])
            except (ValueError, TypeError, AttributeError, RecursionError) as exc:
                status, error = "plan_parse_failed", str(exc)
        new = dict(old, status=status, plan=plan, error=error,
                   normalizer_version=VERSION,
                   normalization_source=str(source / "full/plans.jsonl"))
        audit = dict(sample_id=sid, nested_semantic_keys=nested_keys,
                     old=route(old), new=route(new))
        new["normalization_audit"] = audit
        normalized.append(new)
        audits.append(audit)
    normalized_by_id = {str(row["sample_id"]): row for row in normalized}
    for cohort in ("pilot", "full"):
        cohort_sources = records(destination / cohort / "geometry_inputs.jsonl")
        cohort_ids = [str(row["sample_id"]) for row in cohort_sources]
        write_jsonl(destination / cohort / "plans.jsonl", [normalized_by_id[sid] for sid in cohort_ids])
    audit = dict(
        normalizer_version=VERSION, created_at=datetime.now(timezone.utc).isoformat(),
        source=str(source), output=str(destination), release=str(RELEASE),
        normalizer_sha256=hashlib.sha256(NORMALIZER_FILE.read_bytes()).hexdigest(),
        source_plans_sha256=hashlib.sha256((source / "full/plans.jsonl").read_bytes()).hexdigest(),
        model_generation_reused=True, source_fields_preserved=["raw_text", "messages", "planner_version"],
        gold_used=False, old=counts(audits, "old"), new=counts(audits, "new"),
        nested_plan_records=sum(bool(x["nested_semantic_keys"]) for x in audits),
        newly_accepted_geometry_plans=sum(not x["old"]["route_geometry"] and x["new"]["route_geometry"] for x in audits),
        no_longer_accepted_geometry_plans=sum(x["old"]["route_geometry"] and not x["new"]["route_geometry"] for x in audits),
        newly_feedback_eligible=sum(not x["old"]["feedback_eligible"] and x["new"]["feedback_eligible"] for x in audits),
        no_longer_feedback_eligible=sum(x["old"]["feedback_eligible"] and not x["new"]["feedback_eligible"] for x in audits),
        per_sample=audits,
    )
    audit_path = destination / "plan_reparse_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "per_sample"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
