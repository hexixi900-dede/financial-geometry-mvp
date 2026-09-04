"""Phase 4 legend parsing: OCR series names and extract swatch appearance.

The detector is deliberately rule-based and CPU-only. It groups letter-bearing
OCR tokens into rows, finds a compact colored swatch immediately left (or
right) of a text run, and emits `series_name -> visual appearance` entries
that the series matcher can assign to detected line components.

No pseudo-Gold, value-label boxes, or Gold-adjacent inputs are used.
"""
from __future__ import annotations

import difflib
import math
import re
from typing import Any

import cv2
import numpy as np

from phase3_line_core import normalize_label


def _is_name_token(token: Any, width: int, height: int) -> bool:
    text = str(token.text).strip()
    if len(text) < 2 or re.search(r"[A-Za-z]", text) is None:
        return False
    if token.confidence < 0.35:
        return False
    token_height = token.y2 - token.y1
    token_width = token.x2 - token.x1
    if token_height <= 0 or token_height > 0.08 * height:
        return False
    if token_width <= 0 or token_width > 0.55 * width:
        return False
    # Pure axis-style tokens (years, bare numbers with letters repaired away)
    if re.fullmatch(r"(?i)(fy)?[0-9]{2,4}f?", text):
        return False
    return True


def _swatch_in_zone(image_bgr: np.ndarray, zone: tuple[int, int, int, int]) -> dict[str, Any] | None:
    x1, y1, x2, y2 = zone
    height, width = image_bgr.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 3 or y2 - y1 < 2:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    best: dict[str, Any] | None = None
    for channel_mask, min_pixels, channel in (
        ((hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 250), 8, "saturated"),
        ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 215), 10, "neutral"),
    ):
        mask = channel_mask
        count = int(np.count_nonzero(mask))
        if count < min_pixels:
            continue
        ys, xs = np.nonzero(mask)
        swatch_w = int(xs.max() - xs.min() + 1)
        swatch_h = int(ys.max() - ys.min() + 1)
        # Legend swatches are line segments (wide and thin) or markers (compact).
        if swatch_w < 3 or swatch_h < 2 or swatch_w > 16 * swatch_h:
            continue
        fill = count / float(swatch_w * swatch_h)
        if fill < 0.15:
            continue
        colors = crop[mask]
        if channel == "neutral" and len(colors) >= 8:
            gray = cv2.cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).reshape(-1).astype(float)
            darker = colors[gray <= np.percentile(gray, 60.0)]
            if len(darker) >= 4:
                colors = darker
        median_bgr = np.median(colors, axis=0)
        if channel == "saturated":
            # Hue wraps at 0/179 (red), so use circular concentration instead of std.
            angles = np.deg2rad(hsv[:, :, 0][mask].astype(float) * 2.0)
            resultant = float(np.hypot(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
            if resultant < 0.90:
                continue
        candidate = {
            "bbox": [x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1],
            "median_bgr": [float(value) for value in median_bgr],
            "pixel_count": count,
            "fill": fill,
            "channel": channel,
        }
        if best is None or candidate["pixel_count"] > best["pixel_count"]:
            best = candidate
    return best


def detect_legend_entries(
    image_bgr: np.ndarray,
    tokens: list[Any],
    axis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect legend entries as (swatch, name-run) pairs.

    Returns entries with name, name bbox, swatch appearance, and diagnostics.
    """
    height, width = image_bgr.shape[:2]
    axis_indices = set(int(index) for index in (axis or {}).get("token_indices", []))
    names = [
        token
        for token in tokens
        if token.index not in axis_indices and _is_name_token(token, width, height)
    ]
    if not names:
        return {"entries": [], "candidate_name_tokens": 0, "rows": 0}

    # Group tokens into rows by vertical center.
    names.sort(key=lambda token: (token.cy, token.x1))
    rows: list[list[Any]] = []
    for token in names:
        placed = False
        for row in rows:
            reference = np.median([item.cy for item in row])
            scale = max(np.median([item.y2 - item.y1 for item in row]), 6.0)
            if abs(token.cy - reference) <= 0.6 * scale:
                row.append(token)
                placed = True
                break
        if not placed:
            rows.append([token])
    for row in rows:
        row.sort(key=lambda token: token.x1)
    rows.sort(key=lambda row: min(item.cy for item in row))

    entries: list[dict[str, Any]] = []
    for row in rows:
        row_height = max(float(np.median([item.y2 - item.y1 for item in row])), 6.0)
        current: dict[str, Any] | None = None
        for position, token in enumerate(row):
            gap_left = token.x1 - (row[position - 1].x2 if position else 0)
            # Swatch search zone immediately left of the token.
            zone_left = (
                int(round(token.x1 - 2.6 * row_height)),
                int(round(token.cy - 0.85 * row_height)),
                int(round(token.x1 - 0.12 * row_height)),
                int(round(token.cy + 0.85 * row_height)),
            )
            swatch = None
            if position == 0 or gap_left > 0.35 * row_height:
                swatch = _swatch_in_zone(image_bgr, zone_left)
            if swatch is not None:
                if current is not None:
                    entries.append(current)
                current = {"tokens": [token], "swatch": swatch}
                continue
            if current is not None and gap_left <= 1.5 * row_height:
                current["tokens"].append(token)
                continue
            if current is not None:
                entries.append(current)
                current = None
            # Trailing swatch pattern: "name ----" with swatch to the right.
            zone_right = (
                int(round(token.x2 + 0.12 * row_height)),
                int(round(token.cy - 0.85 * row_height)),
                int(round(token.x2 + 2.6 * row_height)),
                int(round(token.cy + 0.85 * row_height)),
            )
            trailing = _swatch_in_zone(image_bgr, zone_right)
            if trailing is not None:
                current = {"tokens": [token], "swatch": trailing}
        if current is not None:
            entries.append(current)

    output: list[dict[str, Any]] = []
    for entry in entries:
        tokens_in_entry = entry["tokens"]
        name = " ".join(str(token.text).strip() for token in tokens_in_entry)
        name = re.sub(r"\s+", " ", name).strip()
        if len(name) < 2:
            continue
        x1 = min(token.x1 for token in tokens_in_entry)
        y1 = min(token.y1 for token in tokens_in_entry)
        x2 = max(token.x2 for token in tokens_in_entry)
        y2 = max(token.y2 for token in tokens_in_entry)
        output.append(
            {
                "legend_index": len(output),
                "name": name,
                "normalized": normalize_label(name),
                "name_bbox": [float(x1), float(y1), float(x2), float(y2)],
                "name_token_count": len(tokens_in_entry),
                "mean_ocr_confidence": float(np.mean([token.confidence for token in tokens_in_entry])),
                "swatch": entry["swatch"],
            }
        )
    return {
        "entries": output,
        "candidate_name_tokens": len(names),
        "rows": len(rows),
    }


def match_series_query(
    series_query: str,
    legend_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Match a free-text series query to legend entries by rules only."""
    normalized_query = normalize_label(series_query)
    if not normalized_query or not legend_entries:
        return {"status": "no_match", "reason": "empty_query_or_no_legend"}
    scored: list[dict[str, Any]] = []
    for entry in legend_entries:
        normalized_name = str(entry.get("normalized") or "")
        if not normalized_name:
            continue
        containment = 0
        if normalized_name in normalized_query:
            containment = len(normalized_name)
        elif normalized_query in normalized_name:
            containment = len(normalized_query)
        ratio = difflib.SequenceMatcher(None, normalized_query, normalized_name).ratio()
        scored.append(
            {
                "legend_index": int(entry["legend_index"]),
                "legend_name": entry["name"],
                "containment": containment,
                "ratio": ratio,
                "score": containment + ratio,
            }
        )
    if not scored:
        return {"status": "no_match", "reason": "no_usable_legend_name"}
    scored.sort(key=lambda item: (-float(item["score"]), int(item["legend_index"])))
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    accepted = best["containment"] > 0 or best["ratio"] >= 0.72
    return {
        "status": "ok" if accepted else "no_match",
        "match": best,
        "second": second,
        "score_margin": None if second is None else best["score"] - second["score"],
        "legend_entry_count": len(legend_entries),
    }


def strip_query_boilerplate(question: str) -> str:
    """Remove x-axis temporal references and QA boilerplate from a question,
    leaving the series phrase (rule-based first version; no VLM)."""
    text = str(question)
    temporal = [
        r"(?i)\b[1-4]\s*q\s*(?:fy)?\s*[0-9]{2,4}[a-z]?\b",
        r"(?i)\bq\s*[1-4]\s*(?:fy)?\s*[0-9]{2,4}[a-z]?\b",
        r"(?i)\bfy\s*[0-9]{2,4}[a-z]?\b",
        r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[-'/ ]+[0-9]{2,4}[a-z]?\b",
        r"(?i)\b(?:19|20)[0-9]{2}[a-z]?\b",
    ]
    for pattern in temporal:
        text = re.sub(pattern, " ", text)
    text = re.sub(
        r"(?i)\b(what|is|was|were|the|value|of|for|in|on|at|during|approximately|about|"
        r"how|much|did|does|do|reach|reached|report|reported|\?|\.|,)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip(" ?.,")
