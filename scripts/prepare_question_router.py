#!/usr/bin/env python3
"""Prepare full-population, matched-Raw inputs without answer/caption leakage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

VERSION = "question_router_v4"
POPULATION = 11099
SOURCE_KEYS = {
    "sample_id", "chart_id", "image_path", "vlm_image_path", "vlm_image_sha256",
    "question", "question_type", "options", "unit", "input_protocol", "method_version",
}
QUESTION_TYPES = ("single_choice", "multiple_choice", "numerical")

def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def index_unique(records, name):
    result = {str(row["sample_id"]): row for row in records}
    if len(result) != len(records):
        raise ValueError(name + " contains duplicate sample IDs")
    return result

def atomic_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    temporary.replace(path)

def source_from_raw(raw):
    metadata = raw.get("metadata") or {}
    image = str(raw["image_path"])
    digest = raw["image_sha256"]
    result = {
        "sample_id": str(raw["sample_id"]), "chart_id": digest,
        "image_path": image, "vlm_image_path": image, "vlm_image_sha256": digest,
        "question": raw["question"], "question_type": raw["question_type"],
        "options": raw.get("options", metadata.get("options", "")),
        "unit": metadata.get("unit", ""),
        "input_protocol": VERSION, "method_version": VERSION,
    }
    if set(result) != SOURCE_KEYS:
        raise AssertionError("Unexpected inference source fields")
    return result

def select_pilot(all_sources, previous_pilot, previous_candidates):
    """40 established development cases plus 8 unseen-by-structural-filter/type."""
    by_id = index_unique(all_sources, "full source")
    candidate_ids = {str(row["sample_id"]) for row in previous_candidates}
    ordered = sorted(all_sources, key=lambda row: int(row["sample_id"]))
    selected, seen, reasons = [], set(), {}
    def add(sid, reason):
        sid = str(sid)
        if sid not in by_id:
            raise ValueError("Pilot ID outside full population: " + sid)
        if sid not in seen:
            selected.append(by_id[sid])
            seen.add(sid)
            reasons[sid] = reason
    for row in previous_pilot[:40]:
        add(row["sample_id"], "first_40_previous_v4_pilot_development_examples")
    for kind in QUESTION_TYPES:
        candidates = [row for row in ordered
                      if row["sample_id"] not in candidate_ids and row["question_type"] == kind]
        for row in candidates[:8]:
            add(row["sample_id"], "first_8_numeric_ids_outside_old_3699_" + kind)
    # Only handles an unexpected overlap in development IDs; never use answers.
    for row in ordered:
        if len(selected) >= 64:
            break
        add(row["sample_id"], "numeric_id_fill_after_deduplication")
    if len(selected) != 64:
        raise ValueError("Expected exactly 64 distinct development examples")
    return selected, reasons

def prepare(raw_predictions, raw_metrics, old_v4, output):
    raw_predictions, raw_metrics, old_v4, output = map(
        Path, (raw_predictions, raw_metrics, old_v4, output))
    for cohort in ("pilot", "full"):
        for name in ("routers", "outcomes", "plans", "measurements"):
            existing = output / cohort / (name + ".jsonl")
            if existing.exists() and existing.stat().st_size:
                raise RuntimeError("Refusing to replace inputs after experiment work: " + str(existing))
    raw_rows = read_rows(raw_predictions)
    raw = index_unique(raw_rows, "Raw predictions")
    metrics = index_unique(read_rows(raw_metrics), "Raw metrics")
    if len(raw) != POPULATION or set(raw) != set(metrics):
        raise ValueError("Expected matching 11099 Raw prediction and metric IDs")
    sources = [source_from_raw(row) for row in sorted(raw_rows, key=lambda row: int(row["sample_id"]))]
    missing_images = [row["image_path"] for row in sources if not Path(row["image_path"]).is_file()]
    if missing_images:
        raise FileNotFoundError("Original Raw PNG missing; no image substitution: " + repr(missing_images[:5]))
    if any(row["question_type"] not in QUESTION_TYPES for row in sources):
        raise ValueError("Unexpected question type")
    previous_pilot = read_rows(old_v4 / "pilot/geometry_inputs.jsonl")
    previous_candidates = read_rows(old_v4 / "full/geometry_inputs.jsonl")
    if len(previous_candidates) != 3699:
        raise ValueError("Expected previous structural cohort size 3699")
    pilot, reasons = select_pilot(sources, previous_pilot, previous_candidates)
    for cohort, cohort_sources in (("full", sources), ("pilot", pilot)):
        atomic_jsonl(output / cohort / "geometry_inputs.jsonl", cohort_sources)
        # Offline-only reference and baseline files; never merge into source rows.
        atomic_jsonl(output / cohort / "gold.jsonl", [
            {"sample_id": row["sample_id"], "reference": raw[row["sample_id"]]["reference"],
             "tolerance": (raw[row["sample_id"]].get("metadata") or {}).get("tolerance")}
            for row in cohort_sources])
        atomic_jsonl(output / cohort / "baseline.jsonl", [
            {"sample_id": row["sample_id"],
             "raw_vlm_correct": int(bool(metrics[row["sample_id"]]["official_code_correct"]))}
            for row in cohort_sources])
    protocol = {
        "name": VERSION, "input_protocol": VERSION, "method_version": VERSION,
        "source_population": POPULATION, "full_questions": POPULATION, "pilot_questions": len(pilot),
        "post_geometry_raw_fallback": False, "min_pixels": 200704, "max_pixels": 802816,
        "inference_source_fields": sorted(SOURCE_KEYS), "extra_caption": False,
        "gold_available_to_inference": False, "image_policy": "Exact existing image_path PNG used by each Raw record",
        "chart_id": "Raw image_sha256; no old structural chart identifier is substituted",
        "unit_policy": "Only original Raw metadata.unit; no inference or extra metadata",
        "raw_predictions": str(raw_predictions), "raw_metrics": str(raw_metrics),
        "raw_predictions_sha256": hashlib.sha256(raw_predictions.read_bytes()).hexdigest(),
        "raw_metrics_sha256": hashlib.sha256(raw_metrics.read_bytes()).hexdigest(),
        "question_type_counts": dict(Counter(row["question_type"] for row in sources)),
        "pilot_selection": {
            "development_examples": True, "gold_selected": False,
            "description": "First 40 previous v4 pilot IDs, then first 8 numeric sample IDs per question type outside the old 3699 structural candidates; deduplicate and fill by numeric ID to 64.",
            "previous_v4": str(old_v4),
            "sample_ids": [row["sample_id"] for row in pilot], "reasons": reasons,
            "question_type_counts": dict(Counter(row["question_type"] for row in pilot)),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    return protocol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--raw-metrics", required=True)
    parser.add_argument("--old-v4", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = prepare(args.raw_predictions, args.raw_metrics, args.old_v4, args.output)
    print(json.dumps({key: value for key, value in protocol.items() if key != "pilot_selection"}, indent=2))

if __name__ == "__main__":
    main()
