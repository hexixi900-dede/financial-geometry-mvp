"""Render a deterministic post-prediction audit sample; never changes predictions."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import cv2


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--gold",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from finmme_geometry_qa import correctness
    inputs={r["sample_id"]:r for r in rows(args.inputs)}
    gold={r["sample_id"]:r for r in rows(args.gold)}
    pools={"correct":[],"answered_wrong":[],"geometry_refusal":[]}
    for p in rows(args.predictions):
        sid=p["sample_id"]
        if p["status"]=="success":
            g=gold[sid]
            category="correct" if correctness(inputs[sid]["question_type"],p["prediction"],g["reference"],g["tolerance"]) else "answered_wrong"
        elif p.get("stage") in {"line_geometry","bar_geometry","target_grounding"}:
            category="geometry_refusal"
        else:
            continue
        pools[category].append(p)
    selected=[]
    for category,pool in pools.items():
        chosen=[]
        for kind in ("bar","line"):
            chosen += [p for p in pool if p.get("semantic_plan",{}).get("chart_type")==kind][:5]
        for p in pool:
            if len(chosen)>=10:break
            if p not in chosen:chosen.append(p)
        selected += [(category,p) for p in chosen]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for category,p in selected:
        sid=p["sample_id"]
        source=inputs[sid]
        canvas=cv2.imread(source["image_path"])
        h,w=canvas.shape[:2]
        pipelines=p.get("pipeline_audit",[])
        if p.get("bar_audit"):
            pipelines=[p["bar_audit"]]
        for pipeline in pipelines:
            for tick in pipeline.get("axis",{}).get("ticks",[]):
                y=int(round(tick["pixel_y"]))
                cv2.circle(canvas,(int(tick["bbox"][2]),y),3,(255,0,255),-1)
            for anchor in pipeline.get("x_axis_anchors",[]):
                box=anchor.get("bbox")
                if box:cv2.rectangle(canvas,tuple(map(int,box[:2])),tuple(map(int,box[2:])),(255,160,0),1)
            geo=pipeline.get("auto_local_geometry")
            grounding=pipeline.get("grounding",{})
            x=grounding.get("predicted_target_x")
            if x is not None:cv2.line(canvas,(int(x),0),(int(x),h-1),(255,0,255),1)
            if geo:
                wx=geo.get("search_window")
                if wx:cv2.rectangle(canvas,(int(wx[0]),int(pipeline["axis"]["plot_bbox"][1])),(int(wx[1]),int(pipeline["axis"]["plot_bbox"][3])),(0,190,220),1)
                for fit in geo.get("fitted_segments",[]):
                    x1,x2=fit["x_min"],fit["x_max"]
                    y1=fit["target_y"]+fit["slope"]*(x1-geo["target_x"])
                    y2=fit["target_y"]+fit["slope"]*(x2-geo["target_x"])
                    cv2.line(canvas,(int(x1),int(y1)),(int(x2),int(y2)),(0,170,0),2)
                cv2.circle(canvas,tuple(int(round(v)) for v in geo["point"]),6,(0,0,255),2)
        for target in p.get("target_audit",[]):
            if target.get("bbox"):
                x,y,bw,bh=map(int,target["bbox"])
                cv2.rectangle(canvas,(x,y),(x+bw,y+bh),(0,0,255),2)
        plan=p.get("semantic_plan",{})
        lines=[f"{sid} {category}: {p['status']}"]
        lines+=textwrap.wrap(source["question"],max(35,w//8))
        lines+=textwrap.wrap("Targets: "+json.dumps(plan.get("targets",[]),ensure_ascii=True),max(35,w//8))
        lines+=textwrap.wrap(f"Operation={plan.get('operation')} | Geometry={p.get('geometry_values')} | Answer={p.get('prediction')} | Final Gold={gold[sid]['reference']}",max(35,w//8))
        canvas=cv2.copyMakeBorder(canvas,0,20*len(lines)+12,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        for i,line in enumerate(lines):cv2.putText(canvas,line,(6,h+20+i*20),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1,cv2.LINE_AA)
        filename=f"{category}_{sid}.jpg"
        cv2.imwrite(str(args.output_dir/filename),canvas)
        manifest.append({"sample_id":sid,"category":category,"file":filename})
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Rendered {len(selected)} post-prediction overlays")


if __name__=="__main__":main()
