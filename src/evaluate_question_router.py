"""Official, full-population scoring for independent question routing.

Only a terminal route=raw can select a cached Raw answer. Framework failures and
router errors count incorrect. Missing outcomes may use Raw only in explicitly
provisional reports; every final full-population result requires 11099 outcomes.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from vlm_evidence_answerer import official_correct, accuracy_summary

METHOD_VERSION = "question_router_v4"
QUESTION_LABELS = (("single_choice", "Single"), ("multiple_choice", "Multi"), ("numerical", "Calculation"))

def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def unique(rows, label):
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(label + " contains duplicate IDs")
    return result

def needs_geometry(router):
    value = router.get("needs_geometry")
    if type(value) is bool:
        return value
    for key in ("decision", "parsed_reply", "router"):
        nested = router.get(key)
        if isinstance(nested, dict) and type(nested.get("needs_geometry")) is bool:
            return nested["needs_geometry"]
    return None

def measured_value(value):
    if type(value) in (int, float):
        return math.isfinite(value)
    if isinstance(value, dict) and value.get("kind") == "sequence":
        return bool(value.get("samples") or value.get("points") or value.get("values"))
    return False

def measurement_state(outcome, measurement):
    # The runner records the state from evidence actually supplied to the VLM.
    explicit = outcome.get("measurement_state")
    if explicit in {"success", "partial", "none"}:
        return explicit
    explicit = measurement.get("measurement_state")
    if explicit in {"success", "partial", "none"}:
        return explicit
    evidence = measurement.get("evidence") or {}
    values = [item.get("value") for item in evidence.get("measurements", []) if isinstance(item, dict)]
    if not values:
        values = measurement.get("geometry_values") or []
    if not values:
        values = [item.get("auto_local_value") for item in measurement.get("pipeline_audit", [])
                  if isinstance(item, dict) and item.get("status") == "success"]
    available = sum(measured_value(value) for value in values)
    if not available:
        return "none"
    target_count = len((measurement.get("semantic_plan") or {}).get("targets") or [])
    complete = measurement.get("status") == "success" and (not target_count or available >= target_count)
    return "success" if complete else "partial"

def stats(records):
    # Reuse released accuracy bookkeeping while translating only its route label.
    normalized = [dict(row, route="geometry_enhanced" if row["route"] == "framework" else row["route"])
                  for row in records]
    result = accuracy_summary(normalized)
    result["hybrid_correct"] = result["selective_correct"]
    result["hybrid_accuracy"] = result["selective_accuracy"]
    return result

def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)

def evaluate(out, raw_metrics, raw_predictions):
    """Evaluate one pilot/full cohort directory and return its saved summary dict."""
    out, raw_metrics, raw_predictions = map(Path, (out, raw_metrics, raw_predictions))
    inputs = read_rows(out / "geometry_inputs.jsonl")
    sources = unique(inputs, "Cohort source")
    metrics_rows = read_rows(raw_metrics)
    metrics = unique(metrics_rows, "Full Raw metrics")
    raw = unique(read_rows(raw_predictions), "Full Raw predictions")
    if not metrics or set(metrics) != set(raw):
        raise ValueError("Raw predictions must match the full metric population")
    if not set(sources) <= set(metrics):
        raise ValueError("Cohort source IDs outside the Raw population")
    outcomes = unique(read_rows(out / "outcomes.jsonl"), "Terminal outcomes")
    routers = unique(read_rows(out / "routers.jsonl"), "Router journal")
    measurements = unique(read_rows(out / "measurements.jsonl"), "Measurement journal")
    if not set(outcomes) <= set(sources) or not set(routers) <= set(sources):
        raise ValueError("Outcome/router IDs outside this cohort")
    for sid, outcome in outcomes.items():
        if outcome.get("route") not in {"raw", "framework", "router_error"}:
            raise ValueError("Invalid terminal route for " + sid)
        if outcome.get("status") in {"pending", "running"}:
            raise ValueError("Nonterminal record in outcomes for " + sid)
    records = []
    for metric in metrics_rows:
        sid = str(metric["sample_id"])
        raw_row = raw[sid]
        if "prediction" not in raw_row:
            raise ValueError("Raw prediction unavailable for " + sid)
        if "prediction" in metric and metric["prediction"] != raw_row["prediction"]:
            raise ValueError("Raw metric/prediction mismatch for " + sid)
        qtype = metric["question_type"]
        if raw_row.get("question_type", qtype) != qtype:
            raise ValueError("Raw question-type mismatch for " + sid)
        raw_correct = int(bool(metric["official_code_correct"]))
        outcome = outcomes.get(sid)
        pending = outcome is None
        decision = needs_geometry(routers.get(sid, {}))
        if pending:
            route, prediction, correct = "pending", raw_row["prediction"], raw_correct
            status = "pending" if sid in sources else "outside_current_cohort"
            prediction_source = "cached_raw_provisional_only"
        else:
            route, status = outcome["route"], outcome.get("status", "complete")
            if route == "raw":
                # A no-need route, not a measurement/answer failure, authorizes Raw.
                if decision is not False:
                    raise ValueError("Only needs_geometry=false can terminate as raw: " + sid)
                prediction, correct = raw_row["prediction"], raw_correct
                prediction_source = "cached_raw"
            elif route == "router_error":
                prediction, correct = None, 0
                prediction_source = "router_error_no_answer"
            else:
                if decision is not True:
                    raise ValueError("Framework entry requires needs_geometry=true: " + sid)
                prediction = outcome.get("prediction") if status == "success" else None
                tolerance = raw_row.get("tolerance", (raw_row.get("metadata") or {}).get("tolerance"))
                reference = raw_row.get("reference", metric.get("reference"))
                if reference is None:
                    raise ValueError("Offline scoring reference unavailable for " + sid)
                correct = int(prediction is not None and official_correct(qtype, prediction, reference, tolerance))
                prediction_source = "framework"
        state = measurement_state(outcome or {}, measurements.get(sid, {})) if route == "framework" else None
        records.append({
            "sample_id": sid, "question_type": qtype, "is_candidate": sid in sources,
            "needs_geometry": decision, "geometry_entered": route == "framework",
            "route": route, "status": status, "prediction": prediction,
            "prediction_source": prediction_source, "raw_prediction": raw_row["prediction"],
            "raw_correct": raw_correct, "hybrid_correct": correct, "selective_correct": correct,
            "score_provisional": pending, "terminal_outcome_available": not pending,
            "prediction_available": prediction is not None, "raw_prediction_available": True,
            "measurement_state": state,
        })
    cohort = [row for row in records if row["is_candidate"]]
    full = stats(records)
    summary = stats(cohort)
    pending = sum(row["score_provisional"] for row in cohort)
    full_pending = sum(row["score_provisional"] for row in records)
    full_population = len(records) == 11099
    full_scope = set(sources) == set(metrics)
    complete = full_population and full_scope and not full_pending
    provisional = not complete
    router_decisions = [needs_geometry(record) for record in routers.values()]
    route_counts = Counter(row["route"] for row in cohort if not row["score_provisional"])
    router_parse_errors = {
        sid for sid, router in routers.items()
        if needs_geometry(router) is None and router.get("status") not in {None, "pending", "running"}
    } | {sid for sid, outcome in outcomes.items() if outcome["route"] == "router_error"}
    measurement_counts = Counter(row["measurement_state"] for row in cohort if row["route"] == "framework")
    evaluation_status = "complete" if complete else "pilot_complete" if not full_scope and not pending else "partial"
    by_type = {kind: stats([row for row in cohort if row["question_type"] == kind])
               for kind, _ in QUESTION_LABELS}
    full_by_type = {kind: stats([row for row in records if row["question_type"] == kind])
                    for kind, _ in QUESTION_LABELS}
    summary.update(
        method_version=METHOD_VERSION, input_protocol=METHOD_VERSION,
        evaluation_status=evaluation_status, score_provisional=provisional,
        cohort_complete=not pending, cohort_score_provisional=bool(pending),
        completed=len(outcomes), pending=pending, full_pending=full_pending,
        full_questions=len(records), full_finmme_population=full_population,
        full_cohort_scope=full_scope, full_completed=len(outcomes),
        full_raw_correct=full["raw_correct"], full_selective_correct=full["selective_correct"],
        full_hybrid_correct=full["selective_correct"], full_raw_accuracy=full["raw_accuracy"],
        full_selective_accuracy=full["selective_accuracy"], full_hybrid_accuracy=full["selective_accuracy"],
        full_delta_pp=full["delta_pp"], full_rescued=full["rescued"], full_harmed=full["harmed"],
        by_question_type=by_type, full_by_question_type=full_by_type,
        geometry_entered=route_counts["framework"], raw_routed=route_counts["raw"],
        router_error=route_counts["router_error"], router_parse_error=len(router_parse_errors),
        routing_counts=dict(route_counts), evidence_source_counts=dict(Counter(r.get("evidence_source", "unknown") for r in routers.values())), needs_geometry_true=sum(x is True for x in router_decisions),
        needs_geometry_false=sum(x is False for x in router_decisions),
        needs_geometry_unknown=sum(x is None for x in router_decisions),
        router_pending=len(sources) - len(routers),
        measurement_success=measurement_counts["success"], measurement_partial=measurement_counts["partial"],
        measurement_none=measurement_counts["none"],
        measurement_counts={key: measurement_counts[key] for key in ("success", "partial", "none")},
        post_geometry_raw_fallback=False, raw_fallback_policy="disabled_after_framework_entry",
        score_definition="official_code_correct: exact choice sets; absolute per-question numeric tolerance",
        pending_score_definition="Unfinished population members use cached Raw only for provisional scores. Terminal framework failures and router errors count incorrect.",
        overall_definition="Micro accuracy over all 11099 FinMME questions; never average the three type accuracies",
        accuracy_scale="fraction", raw_prediction_source=str(raw_predictions),
        raw_metric_source=str(raw_metrics), full_prediction_exported=True,
        Raw={"correct": full["raw_correct"], "accuracy": full["raw_accuracy"]},
        Hybrid={"correct": full["selective_correct"], "accuracy": full["selective_accuracy"],
                "provisional": provisional},
    )
    summary["main_table"] = []
    for method, key in (("Raw Qwen2.5-VL", "raw_accuracy"), ("Geometry-enhanced Qwen2.5-VL", "selective_accuracy")):
        row = {"method": method, "Overall": full[key], "score_provisional": provisional and key == "selective_accuracy"}
        row.update({label: full_by_type[kind][key] for kind, label in QUESTION_LABELS})
        summary["main_table"].append(row)
    for row in records:
        row["evaluation_status"] = evaluation_status
        row["full_score_provisional"] = provisional
    atomic_write(out / "predictions_final.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    atomic_write(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-metrics", required=True)
    parser.add_argument("--raw-predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.out, args.raw_metrics, args.raw_predictions), indent=2))

if __name__ == "__main__":
    main()
