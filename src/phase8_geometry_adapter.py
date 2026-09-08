"""Structured semantic targets mapped to cached OCR anchors and image pixels."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from geometry_core import interpolate_pixel_to_value
from phase3_line_core import parse_axis_scalar, recommended_window, robust_local_line_point
from phase4_multiline_core import _resolve_target_series, _local_overlap, MAX_LOCAL_OVERLAP_FRACTION
from phase4_series import detect_line_series_components, local_continuity, strip_masks


def detect_series(image: Any, axis: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain colored-series extraction and recover dark strokes without plot spines."""
    import cv2
    import numpy as np
    colored = [s for s in detect_line_series_components(image, axis) if s.get("median_hsv", [0,0,0])[1] >= 60]
    height, width = image.shape[:2]
    left, top, right, bottom = map(lambda v: int(round(v)), axis["plot_bbox"])
    left, top, right, bottom = max(0,left), max(0,top), min(width,right), min(height,bottom)
    crop = image[top:bottom,left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:,:,1] < 60) & (hsv[:,:,2] <= 180)).astype(np.uint8)
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1,max(20,int(.8*(right-left)))),np.uint8))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((max(20,int(.8*(bottom-top))),1),np.uint8))
    clean = dark & (1-cv2.dilate(horizontal|vertical,np.ones((3,3),np.uint8)))
    mask = cv2.morphologyEx(clean,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    neutral=[]
    for i in range(1,n):
        x,y,w,h,area = map(int,stats[i])
        if w < .24*(right-left) or h < 3 or area < 25:
            continue
        component = labels==i
        full = np.zeros((height,width),np.uint8)
        full[top:bottom,left:right][component]=1
        pixels=crop[component & (dark>0)]
        if len(pixels)==0:
            continue
        bgr=np.median(pixels,axis=0)
        neutral.append(dict(series_id=0,bbox=[x+left,y+top,w,h],median_bgr=bgr.tolist(),median_hsv=[0,0,float(np.mean(bgr))],mask=full,coverage_fraction=w/(right-left),component_area=area,column_count=int(np.any(component,axis=0).sum()),fill_ratio=area/(w*h),score=float(w/(right-left)),style_hint="neutral",detector="phase8_dark_stroke_no_spines"))
    # Dark fragments shadowing a colored series are anti-aliasing, not another series.
    neutral=[s for s in neutral if not any(np.count_nonzero((s["mask"]>0)&(cv2.dilate(c["mask"],np.ones((5,5),np.uint8))>0)) > .5*np.count_nonzero(s["mask"]) for c in colored)]
    neutral.sort(key=lambda s: s["bbox"][0])
    merged=[]
    for s in neutral:
        if merged:
            previous=merged[-1]
            end=previous["bbox"][0]+previous["bbox"][2]-1
            start=s["bbox"][0]
            gap=start-end
            y1=np.flatnonzero(previous["mask"][:,end])
            y2=np.flatnonzero(s["mask"][:,start])
            if 0 < gap < .06*width and len(y1) and len(y2) and abs(np.median(y1)-np.median(y2)) < .12*height and np.linalg.norm(np.array(previous["median_bgr"])-s["median_bgr"]) < 35:
                previous["mask"] |= s["mask"]
                ys,xs=np.nonzero(previous["mask"])
                previous.update(bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)], component_area=int(len(xs)),column_count=int(len(np.unique(xs))),merged_fragments=True)
                continue
        merged.append(s)
    neutral=merged
    series=colored+neutral
    for i,s in enumerate(series):s["series_id"]=i
    return series


def scalar(text: str) -> tuple[float, str] | None:
    text = text.strip().replace("’", "/").replace("'", "/")
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%d %B %Y", "%B %d, %Y"):
        try:
            date = datetime.strptime(text, fmt)
            # Match the existing month/quarter fractional-year coordinate convention.
            import calendar
            return date.year + (date.month - 1 + (date.day - 1) / calendar.monthrange(date.year, date.month)[1]) / 12, "temporal"
        except ValueError:
            pass
    parsed = parse_axis_scalar(text)
    if parsed is None:
        return None
    return parsed.value, "temporal" if parsed.family in {"year", "quarter", "month"} else "numeric"


def ground_label(label: str, anchors: list[dict[str, Any]]) -> dict[str, Any]:
    normalize = lambda text: re.sub(r"[^a-z0-9]+", "", text.lower())
    target_scalar = scalar(label)
    exact = []
    compatible = []
    for anchor in anchors:
        value = scalar(str(anchor["label"]))
        if normalize(label) == normalize(str(anchor["label"])):
            exact.append(anchor)
        elif target_scalar and value and target_scalar[1] == value[1]:
            if abs(target_scalar[0] - value[0]) < 1e-6:
                exact.append(anchor)
        if target_scalar and value and target_scalar[1] == value[1]:
            compatible.append((value[0], anchor))
    base = {"target_semantic_label": label}
    if exact:
        if max(float(a["center_x"]) for a in exact) - min(float(a["center_x"]) for a in exact) > 8:
            return {**base, "status": "x_grounding_unrecoverable", "reason": "ambiguous_repeated_x_label"}
        best = max(exact, key=lambda a: float(a.get("confidence", 0)))
        return {**base, "status": "ok", "mode": "direct_anchor", "predicted_target_x": best["center_x"], "matched_detected_label": best["label"], "interpolation_anchors": []}
    compatible.sort(key=lambda pair: pair[0])
    for (left_value, left), (right_value, right) in zip(compatible, compatible[1:]):
        if left_value <= target_scalar[0] <= right_value and right_value > left_value and left["center_x"] < right["center_x"]:
            ratio = (target_scalar[0] - left_value) / (right_value - left_value)
            x = left["center_x"] + ratio * (right["center_x"] - left["center_x"])
            return {**base, "status": "ok", "mode": "piecewise_linear_interpolation", "predicted_target_x": x, "matched_detected_label": None, "interpolation_anchors": [left, right]}
    return {**base, "status": "x_grounding_unrecoverable", "reason": "no_bracketing_semantic_anchors"}


def pixel_region(target, image):
    region = target.get("region")
    if region is None: return None
    height, width = image.shape[:2]
    return [int(round(region[0]*width/1000)), int(round(region[1]*height/1000)),
            min(width, int(round(region[2]*width/1000))), min(height, int(round(region[3]*height/1000)))]


def read_line(image: Any, chart: dict[str, Any], target: dict[str, str], series: list[dict[str, Any]]) -> dict[str, Any]:
    axis = chart["axis"]
    anchors = chart["x_axis_anchors"]
    grounding = ground_label(target["x_label"], anchors)
    resolution = _resolve_target_series(target["series"] or None, chart["legend"], series, "vlm_semantic_series")
    if (resolution["status"] != "ok" and len(series) == 1
            and (chart.get("single_series_hint") or (target["series"] and
                 re.sub(r"[^a-z0-9]", "", target["series"].lower()) in
                 re.sub(r"[^a-z0-9]", "", chart.get("caption", "").lower())))
            and not chart["legend"].get("entries")):
        resolution = {"status": "ok", "target_series_id": series[0]["series_id"],
                      "series_mode": "single_series_without_legend", "semantic_name": target["series"]}
    region = pixel_region(target, image)
    if region:
        import numpy as np
        x1,y1,x2,y2 = region
        if resolution["status"] != "ok":
            scores = sorted([(int(np.count_nonzero(t["mask"][y1:y2,x1:x2])), int(t["series_id"])) for t in series], reverse=True)
            if scores and scores[0][0] > 0 and (len(scores) == 1 or scores[0][0] > scores[1][0]*1.5):
                resolution = {"status":"ok", "target_series_id":scores[0][1], "series_mode":"vlm_visual_region"}
        if grounding["status"] != "ok":
            grounding = {"status":"ok", "mode":"vlm_visual_region", "predicted_target_x":(x1+x2)/2,
                         "target_semantic_label":target["x_label"]}
    selected = None
    selected_mask = None
    search_region = None
    if resolution["status"] == "ok":
        import numpy as np
        selected = next(t for t in series if int(t["series_id"]) == int(resolution["target_series_id"]))
        selected_mask = selected["mask"]
        if region:
            # A detector component can contain crossing strokes of the same
            # color. The VLM region must also select pixels within that component.
            height, width = image.shape[:2]
            padding = max(2, int(round(.01 * min(height, width))))
            x1, y1, x2, y2 = region
            search_region = [max(0, x1-padding), max(0, y1-padding),
                             min(width, x2+padding), min(height, y2+padding)]
            x1, y1, x2, y2 = search_region
            selected_mask = np.zeros_like(selected["mask"])
            selected_mask[y1:y2, x1:x2] = selected["mask"][y1:y2, x1:x2]
    # Endpoint semantics come from the VLM plan; read the selected mark's pixels.
    if target.get("position") in {"first", "last"} and selected_mask is not None:
        import numpy as np
        ys, xs = np.nonzero(selected_mask)
        if len(xs):
            edge_x = int(xs.min() if target["position"] == "first" else xs.max())
            grounding = {"status": "ok", "mode": "vlm_"+target["position"]+"_point",
                         "predicted_target_x": edge_x, "target_semantic_label": target["x_label"]}
    result = {"axis": axis, "x_axis_anchors": anchors, "grounding": grounding, "legend": chart["legend"], "series_resolution": resolution, "detected_series": strip_masks(series)}
    if search_region is not None:
        result["visual_search_region"] = search_region
    if grounding["status"] != "ok":
        return {**result, "status": "x_grounding_unrecoverable", "status_reason": grounding["reason"]}
    if resolution["status"] != "ok":
        return {**result, "status": resolution["status"], "status_reason": resolution.get("reason")}
    if search_region is not None and not selected_mask.any():
        return {**result, "status": "line_localization_failed", "status_reason": "no_series_pixels_in_visual_region"}
    if target.get("measurement") == "sequence":
        import numpy as np
        l,t,r,b = map(int, axis["plot_bbox"])
        if region: l,r = max(l,region[0]),min(r,region[2])
        points=[]
        for px in range(max(0,l), min(image.shape[1],r+1)):
            ys=np.flatnonzero(selected_mask[:,px])
            if len(ys):
                py=float(np.median(ys))
                points.append({"x_pixel":px,"y_pixel":py,"value":interpolate_pixel_to_value(py,axis)})
        if not points:return {**result,"status":"line_localization_failed"}
        vals=[p["value"] for p in points]
        trace={"kind":"sequence", "points":points, "x_axis_anchors":anchors,
               "min":min(vals), "max":max(vals), "first":vals[0], "last":vals[-1],
               "coverage_fraction":len(points)/max(1,r-l+1),
               "note":"Pixel samples, not discrete observations. Do not compute a time average by averaging these samples."}
        return {**result,"status":"success","auto_local_value":trace,
                "auto_local_geometry":{"trace_points":[[p["x_pixel"],p["y_pixel"]] for p in points],"fitted_segments":[]}}
    others = [s["mask"] for s in series if s is not selected]
    x = float(grounding["predicted_target_x"])
    window = recommended_window(x, anchors, image.shape[1])
    overlap = _local_overlap(selected_mask, others, x, window)
    result.update(local_search_window=[x-window, x+window], series_continuity=local_continuity(selected_mask, x, window), series_local_overlap=overlap)
    if others and overlap["local_overlap_fraction"] > MAX_LOCAL_OVERLAP_FRACTION:
        return {**result, "status": "ambiguous_series", "status_reason": "local_overlap_with_other_series"}
    if target.get("position") in {"first", "last"}:
        import numpy as np
        column = np.flatnonzero(selected_mask[:, int(x)])
        if len(column):
            y = float(np.median(column))
            geometry = {"point": [x, y], "target_x": x, "target_y": y,
                        "method": "endpoint_column", "fitted_segments": [], "search_window": [x,x]}
            return {**result, "status": "success", "auto_local_geometry": geometry,
                    "auto_local_value": interpolate_pixel_to_value(y,axis)}
    geometry = robust_local_line_point(selected_mask, x, axis, window)
    if geometry is not None:
        segments = geometry.get("fitted_segments", [])
        if not segments or not (segments[0]["x_min"] <= x <= segments[0]["x_max"]):
            geometry = None
    if geometry is None:
        return {**result, "status": "line_localization_failed", "status_reason": "no_recoverable_point_at_target_x"}
    return {**result, "status": "success", "auto_local_geometry": geometry, "auto_local_value": interpolate_pixel_to_value(float(geometry["target_y"]), axis)}


from phase9_plan_contract import validate_plan


def refresh_anchors(reader: Any, image: Any, chart: dict[str, Any]) -> dict[str, Any]:
    """Recover full dates discarded by the historical month/year-only OCR parser."""
    from phase3_line_core import x_axis_band
    axis = chart["axis"]
    height, width = image.shape[:2]
    x1, y1, x2, y2, baseline = x_axis_band(axis, width, height)
    raw = reader.readtext(image[y1:y2, x1:x2], paragraph=False, canvas_size=2560, mag_ratio=3, min_size=5)
    temporal = []
    for box, text, confidence in raw:
        parsed = scalar(text)
        if confidence < .30 or parsed is None or parsed[1] != "temporal":
            continue
        xs, ys = [float(p[0])+x1 for p in box], [float(p[1])+y1 for p in box]
        temporal.append(dict(label=text, center_x=sum(xs)/4, center_y=sum(ys)/4, bbox=[min(xs),min(ys),max(xs),max(ys)], confidence=float(confidence), source="phase8_upright_date_ocr"))
    if len(temporal) < 2:
        return chart
    # Use the first coherent text row beneath the axis, excluding legend/footnote dates.
    temporal.sort(key=lambda a: a["center_y"])
    first_y = temporal[0]["center_y"]
    temporal = [a for a in temporal if abs(a["center_y"]-first_y) < max(8, .025*height)]
    if len(temporal) < 2:
        return chart
    temporal.sort(key=lambda a: a["center_x"])
    values = [scalar(a["label"])[0] for a in temporal]
    if any(b <= a for a,b in zip(values,values[1:])):
        return chart
    # Preserve useful rotated/month anchors instead of discarding every temporal anchor.
    existing = [a for a in chart["x_axis_anchors"]
                if not any(abs(float(a["center_x"])-b["center_x"]) < 5 for b in temporal)]
    return {**chart, "x_axis_anchors": existing+temporal, "x_axis_repair": "upright_full_date_ocr"}


def _bar_series_color(target, legend):
    """Use an available legend to distinguish a requested bar series or segment."""
    if not target.get("series") or not (legend or {}).get("entries"):
        return None
    from phase4_legend import match_series_query
    entries = legend["entries"]
    match = match_series_query(target["series"], entries)
    if match.get("status") != "ok":
        return None
    entry = next(e for e in entries if e["legend_index"] == match["match"]["legend_index"])
    return entry.get("swatch", {}).get("median_bgr") or entry.get("median_bgr")


def match_bar_target(target, bars, legend=None):
    """Match an explicit category directly, using OCR token positions."""
    normalize = lambda v: re.sub(r"[^a-z0-9]+", "", str(v).lower())
    query = normalize(target["x_label"])
    color = _bar_series_color(target, legend)
    candidates = bars
    if color is not None:
        # A grouped category shares an OCR anchor. Resolve the series before
        # choosing the nearest bar, or every series collapses to the same bar.
        candidates = [b for b in bars if sum((a-c)**2 for a,c in zip(b["median_bgr"],color))**.5 < 75]
    if not candidates:
        return {"status": "semantic_series_not_found"}
    matches = []
    for bar in bars:
        label = normalize(bar.get("detected_x_label", ""))
        tokens = [t for t in bar.get("x_label_tokens", []) if normalize(t.get("text", "")) == query]
        if label == query or tokens:
            for token in tokens or [None]:
                if token is not None:
                    box = token.get("bbox")
                    cx = token.get("cx")
                    if cx is None and box: cx = (box[0]+box[2])/2
                    # A long OCR word-run is not an individual category anchor.
                    if cx is None: continue
                    chosen = min(candidates, key=lambda b: abs(b["target_x"]-cx))
                else:
                    chosen = bar
                if chosen not in candidates:
                    continue
                if chosen not in matches: matches.append(chosen)
    if target.get("series") and legend and legend.get("entries"):
        from phase4_legend import match_series_query
        name_match = match_series_query(target["series"], legend["entries"])
        if name_match.get("status") != "ok":
            return {"status": "semantic_series_not_found"}
    if len(matches) != 1:
        return {"status": "semantic_target_not_found" if not matches else "semantic_target_ambiguous"}
    return {"status": "success", "target_bar_ids": [matches[0]["bar_id"]],
            "semantic_target": target, "method": "structured_category_to_bar"}


def read_bar_height(reader, image, analysis, target):
    """Read the top-bottom extent of a category's rectangular mark, including floating bars."""
    import numpy as np
    from geometry_core import _quantized_components
    axis = analysis['axis']
    left, top, right, bottom = axis['plot_bbox']
    height, width = image.shape[:2]
    norm = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
    query = norm(target['x_label'])
    region = pixel_region(target, image)
    series_color = _bar_series_color(target, analysis.get('legend'))
    baseline = min(height-1, int(round(-axis['intercept']/axis['slope'])))
    baseline = max(int(max(t['pixel_y'] for t in axis['ticks'])), baseline)
    boxes = []
    for comp in _quantized_components(image, axis):
        x,y,w,h = [int(comp[k]) for k in ('x','y','w','h')]
        if not (5 <= w <= .17*width and h >= 2 and comp['fill'] >= .65
                and x >= left-3 and x+w <= right+3 and y >= top-3 and y+h <= bottom+3):
            continue
        if series_color is not None and np.linalg.norm(np.asarray(comp['median_bgr'])-series_color) >= 75:
            continue
        if region:
            rx1,ry1,rx2,ry2 = region
            overlap = max(0,min(x+w,rx2)-max(x,rx1))*max(0,min(y+h,ry2)-max(y,ry1))
            if target.get('measurement') == 'value':
                # A value region may enclose just the endpoint, not half the column.
                cx=(rx1+rx2)/2
                if x-2 <= cx <= x+w+2 and overlap > 0: boxes.append((x,y,w,h))
            elif overlap >= .5*w*h: boxes.append((x,y,w,h))
            continue
        x1,x2=max(0,x-3),min(width,x+w+3)
        y1,y2=max(0,baseline),min(height,baseline+max(20,int(.16*height)))
        if y2<=y1: continue
        words=reader.readtext(image[y1:y2,x1:x2],paragraph=False,detail=0,mag_ratio=2)
        if query and any(norm(word)==query for word in words): boxes.append((x,y,w,h))
    if not boxes:return {'status':'semantic_target_not_found'}
    # Select one column before joining vertically stacked pieces. Never union neighbours.
    columns=[]
    for box in sorted(boxes):
        for col in columns:
            if abs((box[0]+box[2]/2)-(col[0][0]+col[0][2]/2)) <= max(3,min(box[2],col[0][2])*.25):
                col.append(box);break
        else:columns.append([box])
    if len(columns)>1:
        if region is None:return {'status':'ambiguous_bar_columns'}
        cx=(region[0]+region[2])/2
        columns.sort(key=lambda col:abs(col[0][0]+col[0][2]/2-cx))
        distances=[abs(col[0][0]+col[0][2]/2-cx) for col in columns]
        if distances[1]-distances[0]<2:return {'status':'ambiguous_bar_columns'}
    boxes=columns[0]
    # A category's vertical extent can include touching differently colored pieces.
    x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
    right=max(b[0]+b[2]-1 for b in boxes);bottom=max(b[1]+b[3]-1 for b in boxes)
    top_value=interpolate_pixel_to_value(y,axis);bottom_value=interpolate_pixel_to_value(bottom,axis)
    endpoint_y,endpoint_value=(y,top_value) if abs(top_value)>=abs(bottom_value) else (bottom,bottom_value)
    return {'status':'success','bbox':[x,y,right-x+1,bottom-y+1],
            'point':[(x+right)/2,endpoint_y if target.get('measurement')=='value' else y],'bottom_point':[(x+right)/2,bottom],
            'value':abs(top_value-bottom_value) if target.get('measurement') == 'height' else endpoint_value,'top_value':top_value,'bottom_value':bottom_value,
            'measurement':target.get('measurement','height'),'semantic_target':target}
