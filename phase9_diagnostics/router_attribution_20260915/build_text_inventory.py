"""CPU-only OCR text inventory for the router attribution diagnostic.

Records what text a chart actually contains, which tokens the existing axis
calibration already owns as ticks, and whether each token sits inside the plot
area. This decides nothing and calls no model: it only makes later
"printed_values" claims falsifiable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ocr-models", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "src"))
    import cv2
    import easyocr
    import torch
    from geometry_toolbox import prepare_chart

    torch.set_num_threads(8)
    reader = easyocr.Reader(["en"], gpu=False,
                            model_storage_directory=str(args.ocr_models),
                            download_enabled=False, verbose=False)

    requests = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if args.limit:
        requests = requests[: args.limit]

    done = set()
    if args.output.exists():
        done = {json.loads(line)["sample_id"]
                for line in args.output.read_text().splitlines() if line.strip()}

    charts: dict[str, dict] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as out:
        for request in requests:
            sample_id = str(request["sample_id"])
            if sample_id in done:
                continue
            chart_id = request["chart_id"]
            if chart_id not in charts:
                image = cv2.imread(request.get("vlm_image_path") or request["image_path"])
                charts[chart_id] = prepare_chart(reader, image)
            chart = charts[chart_id]

            axes = chart.get("axes") or {}
            tick_indices: set[int] = set()
            ticks = []
            for side, axis in axes.items():
                tick_indices.update(axis.get("token_indices") or [])
                for tick in axis.get("ticks") or []:
                    ticks.append(dict(side=side, text=tick.get("text"), value=tick.get("value"),
                                      pixel_y=tick.get("pixel_y"), bbox=tick.get("bbox")))

            plot = chart.get("plot_bbox")
            texts = []
            for token in chart.get("ocr_tokens") or []:
                center = token.get("center") or [None, None]
                cx, cy = center[0], center[1]
                inside = bool(plot) and plot[0] <= cx <= plot[2] and plot[1] <= cy <= plot[3]
                texts.append(dict(index=token.get("index"), text=token.get("text"),
                                  confidence=token.get("confidence"), bbox=token.get("bbox"),
                                  cx=cx, cy=cy,
                                  numeric_value=token.get("numeric_value"),
                                  numeric_suffix=token.get("numeric_suffix"),
                                  is_axis_tick=token.get("index") in tick_indices,
                                  inside_plot=inside))

            record = dict(sample_id=sample_id, chart_id=chart_id,
                          status=chart.get("status"),
                          capability_status=chart.get("capability_status"),
                          status_reason=chart.get("status_reason"),
                          image_size=chart.get("image_size"),
                          plot_bbox=plot,
                          axis_ticks=ticks,
                          axis_tick_indices=sorted(tick_indices),
                          x_axis_anchors=[a.get("label") for a in (chart.get("x_axis_anchors") or [])],
                          texts=texts)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            labels = [t for t in texts if t["inside_plot"] and not t["is_axis_tick"]
                      and t["numeric_value"] is not None]
            print(json.dumps(dict(sample_id=sample_id, chart_status=record["status"],
                                  tokens=len(texts), ticks=len(ticks),
                                  in_plot_numeric=len(labels))), flush=True)


if __name__ == "__main__":
    main()
