"""Independent question-level decision for whether pixel-to-value recovery is needed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlm_semantic_planner import extract_json


def _chart_messages(row: dict[str, Any], min_pixels: int, max_pixels: int,
                    system: str, instructions: str) -> list[dict[str, Any]]:
    """Use exactly the original Raw-visible image, question, options and existing unit."""
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("missing_question")
    # Matched inputs retain a Geometry thumbnail in image_path and the exact
    # original Raw PNG in vlm_image_path. Do not substitute the thumbnail.
    image_path = row.get("vlm_image_path") or row.get("image_path")
    if not image_path:
        raise ValueError("missing_original_image")
    supplied = {"question": row["question"], "options": row.get("options", ""),
                "unit": row.get("unit") or ""}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "image": Path(image_path).resolve().as_uri(),
             "min_pixels": min_pixels, "max_pixels": max_pixels},
            {"type": "text", "text": instructions + "\n\nActual input:\n" +
             json.dumps(supplied, ensure_ascii=False)},
        ]},
    ]


SOURCES = {"coordinate_reading", "printed_values", "qualitative"}
FORMAT_INSTRUCTION = 'Return only {"evidence_source":"coordinate_reading|printed_values|qualitative","reason":"brief explanation"}; choose ONE evidence_source.'


def messages(row: dict[str, Any], min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    return _chart_messages(row, min_pixels, max_pixels,
        "Classify the source of evidence for THIS QUESTION. Arithmetic alone does not require Geometry. Output JSON.",
        """Inspect the image, question and EVERY option. Decide WHERE the required evidence comes from.
Return one JSON object, for example:
{"evidence_source":"coordinate_reading","reason":"The needed point has no printed value; it must be read against the y axis."}
Choose exactly ONE evidence_source:

printed_values: ALL numbers needed for this question are visibly printed as data labels,
table cells, annotations or other text associated with the requested quantities.
Adding, subtracting, dividing or averaging printed numbers is STILL printed_values.
Qualifying examples: a difference between two labelled donut sectors; the average of
numbers printed in one table row; growth calculated from labelled old/new bars.
In the reason identify the needed printed texts and their associated targets.

coordinate_reading: at least one required number is NOT printed at its target and must
be recovered from a plotted position/height/length/segment using the chart's scale.
An approximate value or option comparison still belongs here if a numerical reading is
needed. An axis tick, a proposed option value, and YOUR estimate are not data labels.
Example: reading a January line value when only the axis ticks contain numbers.
If some required values are labelled but others require positions, use coordinate_reading.

qualitative: the question requires no numerical recovery, such as identifying a title,
legend/category, or a clear qualitative pattern. A request to calculate from printed
numbers belongs to printed_values, not coordinate_reading or qualitative.

This stage does not calculate or answer the question. It classifies evidence needs,
not chart complexity or tool support. Never reject a needed coordinate reading because
of multiple series, two axes, mixed marks or a difficult chart. Do not output a plan,
an answer, or a second boolean decision.""")


def normalize_router(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("router_not_object")
    source = value.get("evidence_source")
    if not isinstance(source, str) or source not in SOURCES:
        raise ValueError("invalid_or_missing_evidence_source")
    reason = value.get("reason", "")
    return {"evidence_source": source, "needs_geometry": source == "coordinate_reading",
            "reason": reason.strip() if isinstance(reason, str) else ""}
