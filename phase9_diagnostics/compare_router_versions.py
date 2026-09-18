"""Compare question_router_v4 with question_router_v5 on the shared 64-question pilot.

No Gold is used for the routing comparison; Gold only enters the final score, which
is read from each release's own summary.json. The printed-label checks reuse the
chart text inventory built for the attribution study.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

AUDITED_EXPECTED = {
    "43": "coordinate_reading", "18": "coordinate_reading", "11095": "coordinate_reading",
    "828": "qualitative", "1960": "qualitative", "1491": "qualitative",
}
LABELS = {"coordinate_reading": "coordinate_reading", "printed_values": "printed_values",
          "qualitative": "qualitative"}


def load(path):
    path = Path(path)
    if not path.exists():
        return {}
    return {str(json.loads(line)["sample_id"]): json.loads(line)
            for line in path.read_text().splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo
    v4 = repo / "phase9_evidence/question_router_v4"
    v5 = repo / "phase9_evidence/question_router_v5"
    inventory = load(args.diagnostics / "text_inventory.jsonl")
    questions = load(v5 / "pilot/geometry_inputs.jsonl")

    versions = {}
    for name, root in (("v4", v4), ("v5", v5)):
        versions[name] = dict(
            routers=load(root / "pilot/routers.jsonl"),
            outcomes=load(root / "pilot/outcomes.jsonl"),
            summary=json.loads((root / "pilot/summary.json").read_text())
            if (root / "pilot/summary.json").exists() else {},
            model=load(root / "pilot/routers_model.jsonl"),
            chart_text=load(root / "pilot/chart_text.jsonl"))

    def numbers(text):
        import re
        return {f"{float(m.group(0)):g}" for m in re.finditer(r"-?\d+(?:\.\d+)?", str(text or ""))}

    def unsupported(records):
        """printed_values claims whose quoted number is not in the chart text."""
        bad = []
        for sid, record in records.items():
            if record.get("evidence_source") != "printed_values":
                continue
            facts = inventory.get(sid)
            if not facts:
                continue
            tick = {f"{float(t['value']):g}" for t in (facts.get("axis_ticks") or [])
                    if t.get("value") is not None}
            allnums = set()
            for token in facts.get("texts") or []:
                if token.get("numeric_value") is not None:
                    allnums.add(f"{float(token['numeric_value']):g}")
                allnums |= numbers(token.get("text"))
            claimed = numbers(record.get("printed_label_text")) or numbers(record.get("reason"))
            if not claimed:
                continue
            if claimed & (allnums - tick):
                continue
            bad.append(sid)
        return sorted(bad)

    def alignment(routers):
        out = {}
        for sid, expected in AUDITED_EXPECTED.items():
            got = (routers.get(sid) or {}).get("evidence_source")
            out[sid] = dict(expected=expected, got=got, ok=got == expected)
        return out

    report = []
    report.append("# question_router_v5 与 v4 的 64 题对比")
    report.append("")
    report.append("同一批 64 题、同一原图与分辨率、同一 Raw 基线。Gold 只用于最后的分数。")
    report.append("")

    report.append("## 1. 分流分布")
    report.append("")
    report.append("| 版本 | printed_values | coordinate_reading | qualitative | 进入 Geometry |")
    report.append("|---|---:|---:|---:|---:|")
    for name in ("v4", "v5"):
        rows = versions[name]["routers"]
        counts = Counter(r.get("evidence_source") for r in rows.values())
        entered = sum(1 for r in rows.values() if r.get("needs_geometry") is True)
        report.append(f"| {name} | {counts.get('printed_values',0)} | "
                      f"{counts.get('coordinate_reading',0)} | {counts.get('qualitative',0)} | {entered} |")
    report.append("")

    report.append("## 2. 与人审结论的分流对齐率")
    report.append("")
    report.append("| 版本 | 一致 | 不一致的题 |")
    report.append("|---|---:|---|")
    alignments = {}
    for name in ("v4", "v5"):
        alignments[name] = alignment(versions[name]["routers"])
        bad = [f"Q{sid}（期望 {v['expected']}，实得 {v['got']}）"
               for sid, v in alignments[name].items() if not v["ok"]]
        report.append(f"| {name} | {sum(v['ok'] for v in alignments[name].values())} / 6 | "
                      f"{'、'.join(bad) or '—'} |")
    report.append("")

    report.append("## 3. 无法被图中文本支持的 printed_values")
    report.append("")
    for name in ("v4", "v5"):
        bad = unsupported(versions[name]["routers"])
        report.append(f"- **{name}**：{len(bad)} 题 — {', '.join('Q'+s for s in bad) or '无'}")
    report.append("")

    v5routers = versions["v5"]["routers"]
    gate = Counter(r.get("gate_verdict") for r in v5routers.values())
    report.append("## 4. v5 的核实闸门活动")
    report.append("")
    report.append("| 闸门结论 | 题数 |")
    report.append("|---|---:|")
    for verdict, count in sorted(gate.items(), key=lambda x: -x[1]):
        report.append(f"| {verdict} | {count} |")
    overrides = [(sid, r.get("model_evidence_source"), r.get("evidence_source"))
                 for sid, r in v5routers.items() if r.get("gate_overrode")]
    report.append("")
    if overrides:
        report.append("被闸门改判的题（模型判断 → 最终）：")
        for sid, before, after in sorted(overrides):
            report.append(f"- Q{sid}: {before} → {after} — {v5routers[sid].get('gate_reason')}")
    else:
        report.append("闸门没有改判任何一题。")
    report.append("")

    kinds = Counter((r.get("answer_kind"), r.get("printed_at_mark"))
                    for r in v5routers.values())
    report.append("## 5. v5 的两阶段判断分布")
    report.append("")
    report.append("| answer_kind | printed_at_mark | 题数 |")
    report.append("|---|---|---:|")
    for (kind, printed), count in sorted(kinds.items(), key=lambda x: -x[1]):
        report.append(f"| {kind} | {printed} | {count} |")
    report.append("")

    report.append("## 6. 逐题分流变化")
    report.append("")
    report.append("| 题号 | v4 | v5 | 闸门 | 题型 | 题目 |")
    report.append("|---|---|---|---|---|---|")
    changed = 0
    for sid in sorted(set(versions["v4"]["routers"]) | set(versions["v5"]["routers"]), key=lambda s: int(s)):
        before = (versions["v4"]["routers"].get(sid) or {}).get("evidence_source")
        after = (versions["v5"]["routers"].get(sid) or {}).get("evidence_source")
        mark = ""
        if before != after:
            changed += 1
            mark = "**"
        verdict = (versions["v5"]["routers"].get(sid) or {}).get("gate_verdict", "")
        question = (questions.get(sid, {}).get("question") or "").replace("|", "/")[:60]
        report.append(f"| {mark}Q{sid}{mark} | {before} | {after} | {verdict} | "
                      f"{questions.get(sid, {}).get('question_type','')} | {question} |")
    report.append("")
    report.append(f"分流发生变化的题：{changed} / {len(v5routers)}")
    report.append("")

    report.append("## 7. 下游得分")
    report.append("")
    for name in ("v4", "v5"):
        summary = versions[name]["summary"]
        if not summary:
            report.append(f"- **{name}**：还没有 summary.json")
            continue
        report.append(f"- **{name}**：{summary.get('questions')} 题，Raw {summary.get('raw_correct')} → "
                      f"系统 {summary.get('hybrid_correct')}；"
                      f"进 Geometry {summary.get('geometry_entered')}、"
                      f"完整 {summary.get('measurement_success')}、部分 {summary.get('measurement_partial')}、"
                      f"无值 {summary.get('measurement_none')}；"
                      f"修对 {summary.get('rescued')}、改错 {summary.get('harmed')}")
    report.append("")

    report.append("## 8. 逐题修对 / 改错（官方口径）")
    report.append("")
    report.append("取自各版本的 `predictions_final.jsonl`，与 `summary.json` 的计数同源。"
                  "这两组题才是分数的实际来源。")
    report.append("")
    deltas = {}
    for name, root in (("v4", v4), ("v5", v5)):
        rows = load(root / "pilot/predictions_final.jsonl")
        measured = load(root / "pilot/measurements.jsonl")
        rescued = [r for r in rows.values() if r["route"] == "framework"
                   and r["raw_correct"] == 0 and r["hybrid_correct"] == 1]
        harmed = [r for r in rows.values() if r["route"] == "framework"
                  and r["raw_correct"] == 1 and r["hybrid_correct"] == 0]
        deltas[name] = dict(rescued=[str(r["sample_id"]) for r in rescued],
                            harmed=[str(r["sample_id"]) for r in harmed])
        report.append(f"**{name}** — 修对 {len(rescued)} · 改错 {len(harmed)}")
        report.append("")
        for label, items in (("修对", rescued), ("改错", harmed)):
            for record in sorted(items, key=lambda r: int(r["sample_id"])):
                sid = str(record["sample_id"])
                state = (measured.get(sid) or {}).get("status")
                question = (questions.get(sid, {}).get("question") or "").replace("|", "/")[:56]
                report.append(f"- {label} Q{sid}：Raw `{record['raw_prediction']}` → "
                              f"系统 `{record['prediction']}`（测量 {state}）— {question}")
        report.append("")

    payload = dict(
        v4=dict(distribution=dict(Counter(r.get("evidence_source")
                                         for r in versions["v4"]["routers"].values())),
                alignment=alignments["v4"], summary=versions["v4"]["summary"],
                unsupported=unsupported(versions["v4"]["routers"])),
        v5=dict(distribution=dict(Counter(r.get("evidence_source")
                                         for r in v5routers.values())),
                alignment=alignments["v5"], summary=versions["v5"]["summary"],
                unsupported=unsupported(v5routers),
                gate=dict(gate), overrides=[dict(sample_id=s, before=b, after=a)
                                            for s, b, a in overrides],
                per_sample={sid: dict(model=r.get("model_evidence_source"),
                                      final=r.get("evidence_source"),
                                      gate=r.get("gate_verdict"),
                                      gate_reason=r.get("gate_reason"),
                                      answer_kind=r.get("answer_kind"),
                                      printed_at_mark=r.get("printed_at_mark"),
                                      printed_label_text=r.get("printed_label_text"),
                                      reason=r.get("reason"))
                            for sid, r in v5routers.items()}),
        changed=changed, deltas=deltas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("COMPARISON.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    args.output.write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
