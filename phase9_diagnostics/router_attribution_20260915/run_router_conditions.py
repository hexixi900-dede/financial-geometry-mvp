"""Router attribution diagnostic: re-run the question-level decision under
ablations. This is a separate diagnostic, not a frozen release, and it never
writes to phase9_evidence or phase9_releases.

Conditions
  c0_options_production  : the exact production messages (options visible)
  c1_blind_production    : identical instructions, options removed
  c2_blind_quote         : c1 plus a verbatim-label proof requirement
  c3_blind_two_stage     : c1 plus the two-question decomposition
                           (needs a numeric value? is it printed at the mark?)

c0 vs c1 isolates option leakage, c1 vs c2 isolates the proof requirement,
c1 vs c3 isolates the decomposition. Options are never removed from c0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONDITIONS = ("c0_options_production", "c1_blind_production", "c2_blind_quote", "c3_blind_two_stage")

QUOTE_TAIL = """

Additionally, if and only if evidence_source is printed_values you must prove it:
copy the EXACT text of that label from the image into "printed_label_text", and
name the mark it is attached to in "attached_to". If you cannot copy such a text
from the image, then evidence_source is NOT printed_values; classify it as
coordinate_reading instead.
Return one JSON object like
{"evidence_source":"printed_values","printed_label_text":"70","attached_to":"January 2021 bar","reason":"brief"}
Set printed_label_text and attached_to to "" whenever evidence_source is not printed_values."""

TWO_STAGE = """Answer TWO separate questions about THIS QUESTION's evidence needs.

Step 1 - does answering this question require a numeric value that is not already
given in the question itself?
  * "which year / quarter / series is the highest, lowest, largest, smallest, or
    closest to X" only asks you to LOCATE an extreme mark and name its category.
    A visual comparison of the plotted marks is enough. This is NOT a numeric
    requirement.
  * "what is the value / how much / how many / what is the difference / what is
    the percentage" DOES require a numeric value.
  * Adding, subtracting or averaging numbers that the question already states is
    not a numeric requirement either.

Step 2 - only if step 1 is yes. For every number you still need, is that exact
number printed IN THE IMAGE attached to the required mark?
  * An axis tick is NOT a printed data label.
  * A number that appears only in the question text is NOT a printed data label.
  * Your own estimate is NOT a printed data label.
  * If some needed numbers are printed at their marks and others are not, answer
    false.

Do not answer the question. Do not output a plan.
Return one JSON object like
{"requires_numeric_value":true,"target_has_printed_label":false,"printed_label_text":"","reason":"brief"}
If requires_numeric_value is false, set target_has_printed_label to null.
If target_has_printed_label is true, copy the exact label text from the image into
printed_label_text and name its mark; otherwise use ""."""

TWO_STAGE_SYSTEM = "Answer two separate questions about the evidence this QUESTION needs. Output JSON."


def build_messages(instructions, row, min_pixels, max_pixels, include_options, system):
    image_path = row.get("vlm_image_path") or row.get("image_path")
    if not image_path:
        raise ValueError("missing_original_image")
    supplied = {"question": row["question"]}
    if include_options:
        supplied["options"] = row.get("options", "")
    supplied["unit"] = row.get("unit") or ""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "image": Path(image_path).resolve().as_uri(),
             "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type": "text", "text": instructions + "\n\nActual input:\n" +
             json.dumps(supplied, ensure_ascii=False)},
        ]},
    ]


def normalize_quote(value):
    if not isinstance(value, dict):
        raise ValueError("router_not_object")
    source = value.get("evidence_source")
    if source not in {"coordinate_reading", "printed_values", "qualitative"}:
        raise ValueError("invalid_or_missing_evidence_source")
    label = value.get("printed_label_text") or ""
    attached = value.get("attached_to") or ""
    return dict(evidence_source=source, needs_geometry=source == "coordinate_reading",
                printed_label_text=label.strip() if isinstance(label, str) else "",
                attached_to=attached.strip() if isinstance(attached, str) else "",
                reason=str(value.get("reason", "")).strip())


def normalize_two_stage(value):
    if not isinstance(value, dict):
        raise ValueError("router_not_object")
    flag = value.get("requires_numeric_value")
    if not isinstance(flag, bool):
        raise ValueError("invalid_requires_numeric_value")
    printed = value.get("target_has_printed_label")
    if flag is False:
        source = "qualitative"
    else:
        if printed is True:
            source = "printed_values"
        elif printed is False or printed is None:
            source = "coordinate_reading"
        else:
            raise ValueError("invalid_target_has_printed_label")
    label = value.get("printed_label_text") or ""
    return dict(evidence_source=source, needs_geometry=source == "coordinate_reading",
                requires_numeric_value=flag, target_has_printed_label=printed,
                printed_label_text=label.strip() if isinstance(label, str) else "",
                reason=str(value.get("reason", "")).strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/data/liu_jun/blind_cssa/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "src"))
    import vlm_need_router as router
    from vlm_semantic_planner import extract_json
    from run_unified_resident import Runtime, invoke

    conditions = [c for c in args.conditions.split(",") if c]
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError("unknown condition: " + condition)

    requests = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if args.limit:
        requests = requests[: args.limit]

    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["sample_id"], row["condition"]))

    runtime = Runtime(args.model, None, 200704, 802816, 10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as out:
        for request in requests:
            sample_id = str(request["sample_id"])
            production = router.messages(request, 200704, 802816)
            production_instructions = production[1]["content"][1]["text"].split("\n\nActual input:\n", 1)[0]
            production_system = production[0]["content"]
            for condition in conditions:
                if (sample_id, condition) in done:
                    continue
                if condition == "c0_options_production":
                    messages = production
                    normalize = router.normalize_router
                    include_options, tokens = True, 384
                elif condition == "c1_blind_production":
                    messages = build_messages(production_instructions, request, 200704, 802816,
                                              False, production_system)
                    normalize = router.normalize_router
                    include_options, tokens = False, 384
                elif condition == "c2_blind_quote":
                    messages = build_messages(production_instructions + QUOTE_TAIL, request,
                                              200704, 802816, False, production_system)
                    normalize = normalize_quote
                    include_options, tokens = False, 512
                else:
                    messages = build_messages(TWO_STAGE, request, 200704, 802816, False,
                                              TWO_STAGE_SYSTEM)
                    normalize = normalize_two_stage
                    include_options, tokens = False, 512

                decision, error, raw = None, None, ""
                for attempt in range(2):
                    raw, error = invoke(runtime, messages, tokens)
                    decision = None
                    if not error:
                        try:
                            decision = normalize(extract_json(raw))
                        except (ValueError, TypeError, AttributeError, RecursionError) as exc:
                            error = str(exc)
                    if decision is not None:
                        break
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": "The output could not be parsed: " + str(error) +
                         '. Return only one valid JSON object with the requested keys.'}]
                record = dict(sample_id=sample_id, chart_id=request["chart_id"], condition=condition,
                              options_visible=include_options,
                              status="success" if decision else "router_error",
                              error=error, raw_text=raw)
                if decision:
                    record.update(decision)
                else:
                    record.update(evidence_source=None, needs_geometry=None, reason="")
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(json.dumps(dict(sample_id=sample_id, condition=condition,
                                      evidence_source=record.get("evidence_source"),
                                      needs_geometry=record.get("needs_geometry"),
                                      status=record["status"])), flush=True)


if __name__ == "__main__":
    main()
