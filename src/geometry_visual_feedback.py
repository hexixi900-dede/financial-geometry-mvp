"""Draw program measurements for one visual correction using the answer VLM.

Design reference: ChartAgent's visual feedback, not its implementation:
https://arxiv.org/html/2510.04514v3#A6.SS3
https://aclanthology.org/2026.acl-long.843/
The overlay contains image geometry and plan targets only, never answer labels.
"""
from __future__ import annotations

import math
from pathlib import Path


PLAN_COLOR = '#e67e22'
MEASURED_COLORS = ('#e11d48', '#2563eb', '#00875a', '#9333ea')


def _coordinates(value, count):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        return None
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(v) for v in result) else None


def render_measurement_overlay(source, measurement, output_path):
    """Return an absolute PNG path, preserving the Geometry JPEG coordinates.

    Orange dashed boxes are VLM requests. Solid numbered marks are what the
    program actually located, including partial locations from failed reads.
    """
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(source['image_path']) as original:
        image = original.convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plan = measurement.get('semantic_plan') or {}
    pipes = measurement.get('pipeline_audit') or []
    bar_audit = measurement.get('bar_audit') or {}
    audits = measurement.get('target_audit') or []

    def label(point, text, color):
        x, y = point
        bounds = draw.textbbox((0, 0), text, font=font)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(image.width - width - 4, x))
        y = max(0, min(image.height - height - 4, y))
        draw.rectangle((x, y, x + width + 4, y + height + 4), fill='white')
        draw.text((x + 2, y + 1 - bounds[1]), text, font=font, fill=color)

    def marker(point, color):
        point = _coordinates(point, 2)
        if point:
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline='white', width=1)
        return point

    def dashed_box(box):
        left, top, right, bottom = box
        for start in range(int(left), int(right) + 1, 12):
            end = min(start + 6, right)
            draw.line((start, top, end, top), fill=PLAN_COLOR, width=2)
            draw.line((start, bottom, end, bottom), fill=PLAN_COLOR, width=2)
        for start in range(int(top), int(bottom) + 1, 12):
            end = min(start + 6, bottom)
            draw.line((left, start, left, end), fill=PLAN_COLOR, width=2)
            draw.line((right, start, right, end), fill=PLAN_COLOR, width=2)

    # Calibration comes from the parser, not the VLM's approximate regions.
    axes = [p.get('axis') or {} for p in pipes]
    if bar_audit.get('axis'):
        axes.append(bar_audit['axis'])
    if not axes and measurement.get('y_axis_ticks'):
        axes.append({'ticks': measurement['y_axis_ticks']})
    drawn_axes = set()
    for axis in axes:
        bbox = _coordinates(axis.get('plot_bbox'), 4)
        identity = repr(axis)
        if identity in drawn_axes:
            continue
        drawn_axes.add(identity)
        if bbox:
            draw.rectangle(bbox, outline='#64748b', width=1)
        for tick in axis.get('ticks') or []:
            point = _coordinates((0, tick.get('pixel_y')), 2)
            if point:
                left, right = (bbox[0], bbox[2]) if bbox else (0, image.width - 1)
                draw.line((left, point[1], right, point[1]), fill='#c59acb', width=1)
    for pipe in pipes:
        for anchor in pipe.get('x_axis_anchors') or []:
            bbox = _coordinates(anchor.get('bbox'), 4)
            if bbox:
                draw.rectangle(bbox, outline='#d6a524', width=1)

    for i, target in enumerate(plan.get('targets') or []):
        region = _coordinates(target.get('region'), 4)
        if region and region[0] < region[2] and region[1] < region[3]:
            left, top, right, bottom = (min(1000, max(0, v)) for v in region)
            box = (left * image.width / 1000, top * image.height / 1000,
                   right * image.width / 1000, bottom * image.height / 1000)
            dashed_box(box)
            label((box[0], box[1] - 15), f'T{i} plan', PLAN_COLOR)

    for i in range(max(len(pipes), len(audits))):
        color = MEASURED_COLORS[i % len(MEASURED_COLORS)]
        pipe = pipes[i] if i < len(pipes) else {}
        audit = audits[i] if i < len(audits) else {}
        geometry = pipe.get('auto_local_geometry') or audit.get('geometry') or {}
        target_point = None
        bbox = _coordinates(audit.get('bbox'), 4)
        if bbox:
            x, y, width, height = bbox  # Existing bar audit stores xywh.
            if width > 0 and height > 0:
                draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
                target_point = (x, y)
        trace = [_coordinates(point, 2) for point in geometry.get('trace_points') or []]
        trace = [point for point in trace if point is not None]
        if trace:
            # Every trace entry is a recovered image column. Joining across a
            # missing column would draw evidence the parser never recovered.
            segments = [[trace[0]]]
            for point in trace[1:]:
                if 0 < point[0] - segments[-1][-1][0] <= 1:
                    segments[-1].append(point)
                else:
                    segments.append([point])
            for segment in segments:
                if len(segment) > 1:
                    draw.line(segment, fill=color, width=3)
                else:
                    marker(segment[0], color)
            target_point = trace[len(trace) // 2]
        for fit in geometry.get('fitted_segments') or []:
            parameters = _coordinates((fit.get('x_min'), fit.get('x_max'),
                                       fit.get('target_y'), fit.get('slope'), geometry.get('target_x')), 5)
            if parameters:
                x1, x2, y, slope, tx = parameters
                draw.line((x1, y + slope * (x1 - tx), x2, y + slope * (x2 - tx)), fill=color, width=2)
        point = marker(geometry.get('point') or audit.get('point'), color)
        bottom = marker(audit.get('bottom_point'), color)
        if point and bottom:
            draw.line((*point, *bottom), fill=color, width=2)
        if point:
            target_point = point
        if target_point:
            label((target_point[0] + 7, target_point[1] + 7), f'T{i} measured', color)

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format='PNG')
    return str(output)


def build_feedback_messages(source, measurement, overlay_path, min_pixels, max_pixels, allow_revision=True, no_raw_fallback=False):
    """Give the same answer VLM the original image and the actual measurement map."""
    from vlm_evidence_answerer import build_prompt

    prompt = build_prompt(source, measurement, allow_revision=allow_revision, no_raw_fallback=no_raw_fallback)
    original = Path(source.get('vlm_image_path') or source['image_path']).resolve()
    overlay = Path(overlay_path).resolve()
    content = [
        {'type': 'text', 'text': 'Image 1: original chart used by the Raw VLM.'},
        {'type': 'image', 'image': original.as_uri(), 'min_pixels': min_pixels, 'max_pixels': max_pixels},
        {'type': 'text', 'text': 'Image 2: program measurement overlay on an uncropped chart copy. Orange dashed Tn plan boxes are only the VLM request. Solid Tn measured marks/lines are actual program locations. Gaps between measured line segments are missing samples, not interpolated evidence. Purple lines are axis calibration; yellow boxes are OCR anchors. Missing solid marks mean no location was recovered. Compare positions using the chart, not absolute pixel sizes between the two images.'},
        {'type': 'image', 'image': overlay.as_uri(), 'min_pixels': min_pixels, 'max_pixels': max_pixels},
        {'type': 'text', 'text': prompt},
    ]
    system = ('Use the original chart, question, options and valid geometry evidence to answer. '
              'Distinguish actual measurements from visual estimates. Return one JSON object.'
              if no_raw_fallback else 'Inspect visual measurement feedback and use only supported measured values. Return one JSON object.')
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': content}]
