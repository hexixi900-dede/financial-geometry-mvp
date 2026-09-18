"""Generic pilot comparison between two question_router releases.

Usage: compare_pair.py --repo <repo> --from v9 --to v10
Writes phase9_evidence/question_router_<to>/COMPARISON_<from>_<to>.md
"""
import argparse
import collections
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path("/data/liu_jun/financial_geometry_mvp"))
parser.add_argument("--from", dest="before", required=True)
parser.add_argument("--to", dest="after", required=True)
parser.add_argument("--method", default="")
args = parser.parse_args()

REPO = args.repo
ROOTS = {name: REPO / f"phase9_evidence/question_router_{name}/pilot"
         for name in (args.before, args.after)}
OUT = REPO / f"phase9_evidence/question_router_{args.after}/COMPARISON_{args.before}_{args.after}.md"
B, A = args.before, args.after


def load(root, name):
    try:
        return {str(json.loads(l)["sample_id"]): json.loads(l)
                for l in (root / name).read_text().splitlines() if l.strip()}
    except FileNotFoundError:
        return {}


lines = [f"# question_router {B} → {A} 对比", ""]
if args.method:
    lines += [args.method, ""]

summaries = {}
lines += ["## 总分", ""]
for name in (B, A):
    try:
        summary = json.loads((ROOTS[name] / "summary.json").read_text())
    except FileNotFoundError:
        lines.append(f"- **{name}**：还没有 summary.json")
        continue
    summaries[name] = summary
    lines.append(
        f"- **{name}**：{summary.get('questions')} 题，Raw {summary.get('raw_correct')} → "
        f"系统 {summary.get('hybrid_correct')}；进 Geometry {summary.get('geometry_entered')}、"
        f"完整 {summary.get('measurement_success')}、部分 {summary.get('measurement_partial')}、"
        f"无值 {summary.get('measurement_none')}；修对 {summary.get('rescued')}、"
        f"改错 {summary.get('harmed')}")
if len(summaries) == 2:
    delta = (summaries[A].get("hybrid_correct") or 0) - (summaries[B].get("hybrid_correct") or 0)
    rawdelta = (summaries[A].get("raw_correct") or 0) - (summaries[B].get("raw_correct") or 0)
    lines += ["", f"- 变化：系统 {delta:+d}，Raw {rawdelta:+d}"]
lines.append("")

pb, pa = load(ROOTS[B], "predictions_final.jsonl"), load(ROOTS[A], "predictions_final.jsonl")
keys = ["route", "raw_prediction", "prediction", "raw_correct", "hybrid_correct", "status"]
changed = [(sid, pb.get(sid, {}), pa.get(sid, {})) for sid in sorted(set(pb) | set(pa), key=int)
           if any(pb.get(sid, {}).get(k) != pa.get(sid, {}).get(k) for k in keys)]
mb, ma = load(ROOTS[B], "measurements.jsonl"), load(ROOTS[A], "measurements.jsonl")
lines += [f"## 逐题变化（{len(changed)} 题）", ""]
if changed:
    lines += [f"| 题 | 路径 | {B} 答案 | {A} 答案 | {B} 对 | {A} 对 | 测量成功/总 |",
              "|---|---|---|---|---|---|---|"]
    for sid, a, b in changed:
        lines.append("| Q%s | %s→%s | %s | %s | %s | %s | %s→%s |" % (
            sid, a.get("route"), b.get("route"), a.get("prediction"), b.get("prediction"),
            a.get("hybrid_correct"), b.get("hybrid_correct"),
            f"{mb.get(sid, {}).get('measurement_success_count')}/{mb.get(sid, {}).get('measurement_target_count')}",
            f"{ma.get(sid, {}).get('measurement_success_count')}/{ma.get(sid, {}).get('measurement_target_count')}"))
else:
    lines.append("（无）")
lines.append("")

lines += ["## 测量状态分布", ""]
for name in (B, A):
    rows = load(ROOTS[name], "measurements.jsonl")
    lines.append(f"- **{name}**：{dict(collections.Counter(r.get('status') for r in rows.values()))}")
lines.append("")

repaired = sorted([sid for sid, r in ma.items() if r.get("x_axis_repair")], key=int)
if repaired:
    lines += [f"## {A} 触发轴带重读的题（{len(repaired)} 题）", ""]
    lines += [f"- Q{sid}：{ma[sid]['x_axis_repair']}" for sid in repaired]
    lines.append("")

fell_back = []
for sid, r in ma.items():
    for t in (r.get("target_audit") or []):
        if isinstance(t, dict) and t.get("locator_fallback"):
            fell_back.append((sid, t.get("x_label"), t.get("locator_fallback_offset_pixels")))
if fell_back:
    lines += [f"## {A} 使用最近采样列回退的目标（{len(fell_back)} 个）", "",
              "| 题 | 目标 x | 偏移(px) |", "|---|---|---:|"]
    lines += [f"| Q{sid} | {label} | {offset} |" for sid, label, offset in
              sorted(fell_back, key=lambda x: int(x[0]))]
    lines.append("")

single = []
for sid, r in ma.items():
    if (r.get("semantic_plan") or {}).get("operation") == "evidence" and r.get("geometry_values"):
        single.append((sid, r.get("geometry_values")))
if single:
    lines += [f"## {A} 中 operation=evidence 但有测量值的题（{len(single)} 题）", "",
              "这些题的数字由 Python 产出，选项映射在 v10 起会使用它。", "",
              "| 题 | 测量值 |", "|---|---|"]
    lines += [f"| Q{sid} | {values} |" for sid, values in sorted(single, key=lambda x: int(x[0]))]
    lines.append("")

candidates = load(ROOTS[A], "chart_text_candidates.jsonl")
if candidates:
    lines += [f"## {A} 提供给计划层的图上文本（{len(candidates)} 题）", ""]
    plans = load(ROOTS[A], "plans.jsonl")
    lines += ["| 题 | x 类别 | 图例系列 | 其它文本 | 计划填的 series |", "|---|---|---|---|---|"]
    for sid in sorted(candidates, key=int):
        row = candidates[sid]
        plan = (plans.get(sid, {}).get("plan") or {})
        filled = sorted({str(t.get("series")) for t in (plan.get("targets") or []) if t.get("series")})
        lines.append("| Q%s | %s | %s | %s | %s |" % (
            sid, ", ".join(row.get("x_labels") or [])[:60],
            ", ".join(row.get("legend_series") or [])[:60],
            ", ".join(row.get("other_text") or [])[:60],
            ", ".join(filled)[:60]))
    lines.append("")

OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
