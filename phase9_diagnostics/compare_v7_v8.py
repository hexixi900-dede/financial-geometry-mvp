"""v7 -> v8 comparison over the pilot cohort, plus the per-target locator view."""
import json
from pathlib import Path

REPO = Path("/data/liu_jun/financial_geometry_mvp")
E7 = REPO / "phase9_evidence/question_router_v7/pilot"
E8 = REPO / "phase9_evidence/question_router_v8/pilot"
OUT = REPO / "phase9_evidence/question_router_v8/COMPARISON_v7_v8.md"


def load(path):
    try:
        return {str(json.loads(l)["sample_id"]): json.loads(l)
                for l in path.read_text().splitlines() if l.strip()}
    except FileNotFoundError:
        return {}


lines = ["# question_router v7 → v8 对比", "",
         "v8 只改 x 定位（日期精度锚点、轴带重读、极值全序列），路由与闸门不变。", ""]

summaries = {}
for name, root in (("v7", E7), ("v8", E8)):
    summary = json.loads((root / "summary.json").read_text())
    summaries[name] = summary
    lines.append(f"- **{name}**：{summary.get('questions')} 题，Raw {summary.get('raw_correct')} → "
                 f"系统 {summary.get('hybrid_correct')}；进 Geometry {summary.get('geometry_entered')}、"
                 f"完整 {summary.get('measurement_success')}、部分 {summary.get('measurement_partial')}、"
                 f"无值 {summary.get('measurement_none')}；修对 {summary.get('rescued')}、"
                 f"改错 {summary.get('harmed')}")
lines.append("")

p7, p8 = load(E7 / "predictions_final.jsonl"), load(E8 / "predictions_final.jsonl")
keys = ["route", "raw_prediction", "prediction", "raw_correct", "hybrid_correct", "status"]
changed = [(sid, p7.get(sid, {}), p8.get(sid, {})) for sid in sorted(set(p7) | set(p8), key=int)
           if any(p7.get(sid, {}).get(k) != p8.get(sid, {}).get(k) for k in keys)]
lines += [f"## 逐题变化（{len(changed)} 题）", ""]
if changed:
    lines += ["| 题 | 路径 | v7 答案 | v8 答案 | v7 对 | v8 对 | 测量成功/总 |",
              "|---|---|---|---|---|---|---|"]
    m7, m8 = load(E7 / "measurements.jsonl"), load(E8 / "measurements.jsonl")
    for sid, a, b in changed:
        lines.append("| Q%s | %s→%s | %s | %s | %s | %s | %s→%s |" % (
            sid, a.get("route"), b.get("route"), a.get("prediction"), b.get("prediction"),
            a.get("raw_correct"), b.get("hybrid_correct"),
            f"{m7.get(sid, {}).get('measurement_success_count')}/{m7.get(sid, {}).get('measurement_target_count')}",
            f"{m8.get(sid, {}).get('measurement_success_count')}/{m8.get(sid, {}).get('measurement_target_count')}"))
else:
    lines.append("（无）")
lines.append("")

m8 = load(E8 / "measurements.jsonl")
repaired = [sid for sid, r in m8.items() if r.get("x_axis_repair")]
lines += [f"## v8 触发轴带重读的题（{len(repaired)} 题）", ""]
lines += [f"- Q{sid}：{m8[sid]['x_axis_repair']}" for sid in sorted(repaired, key=int)] or ["（无）"]
lines.append("")

seq = [(sid, r) for sid, r in m8.items()
       if any(t.get("measurement") == "sequence" and not str(t.get("x_label") or "").strip()
              for t in ((r.get("semantic_plan") or {}).get("targets") or []))]
lines += [f"## v8 中按全序列扫描的极值目标（{len(seq)} 题）", ""]
lines += [f"- Q{sid}：operation={(r.get('semantic_plan') or {}).get('operation')}，"
          f"测量状态={r.get('status')}（{r.get('measurement_success_count')}/"
          f"{r.get('measurement_target_count')}）" for sid, r in sorted(seq, key=lambda x: int(x[0]))] or ["（无）"]
lines.append("")

OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
