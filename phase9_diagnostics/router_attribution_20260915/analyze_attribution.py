"""Attribute the router's routing errors using the OCR text inventory.

No Gold is read. A printed_values claim is checked against the text the chart
actually contains and against the tokens the existing axis calibration already
owns, so the claim is falsifiable rather than self-reported.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text) -> set[str]:
    if text is None:
        return set()
    return {m.group(0) for m in NUMBER.finditer(str(text))}


def canon(values) -> set[str]:
    out = set()
    for value in values:
        try:
            out.add(f"{float(value):g}")
        except (TypeError, ValueError):
            out.add(str(value))
    return out


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--frozen-routers", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = {row["sample_id"]: row for row in load_jsonl(args.inventory)}
    conditions = load_jsonl(args.conditions)
    frozen = {row["sample_id"]: row for row in load_jsonl(args.frozen_routers)}
    questions = {row["sample_id"]: row for row in load_jsonl(args.inputs)}
    review = json.loads(args.review.read_text()).get("cases", {})

    chart_facts = {}
    for sample_id, row in inventory.items():
        tokens = row.get("texts") or []
        in_plot = [t for t in tokens if t.get("inside_plot") and not t.get("is_axis_tick")]
        calibrated = row.get("plot_bbox") is not None and bool(row.get("axis_ticks"))
        tick_numbers = canon([t.get("value") for t in (row.get("axis_ticks") or [])])
        tick_numbers |= canon(numbers_in(" ".join(str(t.get("text") or "")
                                                   for t in (row.get("axis_ticks") or []))))
        all_numbers = canon([t.get("numeric_value") for t in tokens
                             if t.get("numeric_value") is not None])
        all_numbers |= canon(numbers_in(" ".join(str(t.get("text") or "") for t in tokens)))
        # A right-hand axis whose fit failed still looks like in-plot numbers. Flag
        # numerics that hug a plot edge and share a column with another numeric.
        plot = row.get("plot_bbox")
        edge_numbers: set[str] = set()
        if plot:
            width = float(plot[2]) - float(plot[0])
            left_edge = float(plot[0]) + 0.12 * width
            right_edge = float(plot[2]) - 0.12 * width
            near = []
            for token in in_plot:
                if token.get("numeric_value") is None:
                    continue
                cx = float(token.get("cx") or 0)
                if cx <= left_edge or cx >= right_edge:
                    near.append(token)
            for token in near:
                column = [other for other in near
                          if abs(float(other.get("cx") or 0) - float(token.get("cx") or 0)) < 0.02 * width]
                if len(column) >= 2:
                    edge_numbers |= canon([token["numeric_value"]])
        chart_facts[sample_id] = dict(
            chart_status=row.get("status"),
            status_reason=row.get("status_reason"),
            calibrated=calibrated,
            tokens=len(tokens),
            axis_ticks=len(row.get("axis_ticks") or []),
            all_numeric=sorted(all_numbers),
            tick_numeric=sorted(tick_numbers),
            non_tick_numeric=sorted(all_numbers - tick_numbers),
            edge_like_numeric=sorted(edge_numbers),
            in_plot_numeric=sorted(canon(
                [t.get("numeric_value") for t in in_plot if t.get("numeric_value") is not None]
                + list(numbers_in(" ".join(str(t.get("text") or "") for t in in_plot))))),
        )

    per_condition = defaultdict(dict)
    for row in conditions:
        per_condition[row["condition"]][row["sample_id"]] = row

    def claim_verdict(sample_id: str, record: dict) -> dict:
        facts = chart_facts.get(sample_id)
        if facts is None:
            return dict(verdict="no_inventory")
        claimed_text = record.get("printed_label_text") or record.get("reason")
        claimed = canon(numbers_in(claimed_text))
        verdict: list[str] = []
        if not claimed:
            verdict.append("no_number_claimed")
        elif claimed & set(facts["non_tick_numeric"]):
            if claimed & set(facts["edge_like_numeric"]):
                verdict.append("likely_uncalibrated_axis_tick")
            else:
                verdict.append("number_exists_in_chart_text")
                if facts["calibrated"] and (claimed & set(facts["in_plot_numeric"])):
                    verdict.append("located_in_plot")
                elif facts["calibrated"]:
                    verdict.append("not_in_plot")
        elif claimed & set(facts["tick_numeric"]):
            verdict.append("only_axis_tick")
        elif claimed & set(facts["all_numeric"]):
            verdict.append("only_axis_tick")
        else:
            verdict.append("absent_from_image")
        options_numbers = canon(numbers_in(questions.get(sample_id, {}).get("options")))
        question_numbers = canon(numbers_in(questions.get(sample_id, {}).get("question")))
        if claimed and options_numbers and (claimed & options_numbers):
            verdict.append("number_present_in_options")
        if claimed and question_numbers and (claimed & question_numbers):
            verdict.append("number_present_in_question")
        if not facts["non_tick_numeric"]:
            verdict.append("chart_has_no_non_tick_number")
        return dict(verdict="|".join(verdict), claimed=sorted(claimed),
                    chart_non_tick_numbers=facts["non_tick_numeric"][:12],
                    chart_tick_numbers=facts["tick_numeric"][:12],
                    calibrated=facts["calibrated"],
                    chart_status=facts["chart_status"])

    conditions_order = [c for c in ("c0_options_production", "c1_blind_production",
                                    "c2_blind_quote", "c3_blind_two_stage")
                        if c in per_condition]

    summary = {}
    for condition in conditions_order:
        rows = per_condition[condition]
        sources = Counter(r.get("evidence_source") for r in rows.values())
        needs = Counter(r.get("needs_geometry") for r in rows.values())
        printed = {sid: r for sid, r in rows.items() if r.get("evidence_source") == "printed_values"}
        verdicts = Counter()
        unsupported = []
        for sid, record in printed.items():
            verdict = claim_verdict(sid, record)
            verdicts[verdict.get("verdict")] += 1
            text = str(verdict.get("verdict"))
            if "absent_from_image" in text or "only_axis_tick" in text or "likely_uncalibrated_axis_tick" in text:
                unsupported.append(sid)
        summary[condition] = dict(
            n=len(rows), evidence_source=dict(sources), needs_geometry=dict(needs),
            printed_values=len(printed), verdicts=dict(verdicts),
            unsupported_printed_claims=sorted(unsupported))

    # c0 fidelity against the frozen release
    fidelity = dict(matched=0, mismatched=[], missing=[])
    for sid, record in per_condition.get("c0_options_production", {}).items():
        reference = frozen.get(sid)
        if reference is None:
            fidelity["missing"].append(sid)
        elif reference.get("evidence_source") == record.get("evidence_source"):
            fidelity["matched"] += 1
        else:
            fidelity["mismatched"].append(dict(
                sample_id=sid, frozen=reference.get("evidence_source"),
                rerun=record.get("evidence_source")))

    flips = {}
    for condition in conditions_order:
        if condition == "c0_options_production":
            continue
        base = per_condition.get("c0_options_production", {})
        rows = per_condition[condition]
        changed = []
        for sid, record in rows.items():
            before = base.get(sid, {}).get("evidence_source")
            after = record.get("evidence_source")
            if before != after:
                changed.append(dict(sample_id=sid, before=before, after=after))
        flips[condition] = dict(changed=len(changed), detail=changed)

    audited = {}
    for sid, note in review.items():
        audited[sid] = dict(note=note.get("finding"), question=questions.get(sid, {}).get("question"),
                            frozen=frozen.get(sid, {}).get("evidence_source"),
                            chart=chart_facts.get(sid), conditions={
                                condition: dict(
                                    source=per_condition[condition].get(sid, {}).get("evidence_source"),
                                    printed_label=per_condition[condition].get(sid, {}).get("printed_label_text", ""),
                                    requires_numeric=per_condition[condition].get(sid, {}).get("requires_numeric_value"),
                                    has_printed=(per_condition[condition].get(sid, {}).get("target_has_printed_label")),
                                    reason=per_condition[condition].get(sid, {}).get("reason", ""),
                                    verdict=claim_verdict(sid, per_condition[condition].get(sid, {})).get("verdict"),
                                ) for condition in conditions_order if sid in per_condition[condition]})

    # Expected routing for the human-audited cases. These come from the image audit
    # written by the project (router_review.json), not from Gold or from a score.
    AUDITED_EXPECTED = {
        "43": "coordinate_reading",     # needs a value read off the axis
        "18": "coordinate_reading",
        "11095": "coordinate_reading",
        "828": "qualitative",           # which year is highest: visual comparison
        "1960": "qualitative",          # which quarter is highest: visual comparison
        "1491": "qualitative",          # which quarters are closest to the lowest
    }
    alignment = {}
    for condition in conditions_order:
        rows = per_condition[condition]
        matched, mismatched = [], []
        for sid, expected in AUDITED_EXPECTED.items():
            got = (rows.get(sid) or {}).get("evidence_source")
            (matched if got == expected else mismatched).append(
                dict(sample_id=sid, expected=expected, got=got))
        alignment[condition] = dict(matched=len(matched), total=len(AUDITED_EXPECTED),
                                    mismatch=mismatched)

    per_sample = {}
    for sample_id, facts in chart_facts.items():
        entry = dict(chart=dict(
            chart_status=facts["chart_status"], status_reason=facts["status_reason"],
            calibrated=facts["calibrated"], tokens=facts["tokens"], axis_ticks=facts["axis_ticks"],
            non_tick_numeric=facts["non_tick_numeric"][:14],
            edge_like_numeric=facts["edge_like_numeric"][:14],
            in_plot_numeric=facts["in_plot_numeric"][:14]), conditions={})
        for condition in conditions_order:
            record = per_condition.get(condition, {}).get(sample_id)
            if not record:
                continue
            entry["conditions"][condition] = dict(
                evidence_source=record.get("evidence_source"),
                needs_geometry=record.get("needs_geometry"),
                requires_numeric_value=record.get("requires_numeric_value"),
                target_has_printed_label=record.get("target_has_printed_label"),
                printed_label_text=record.get("printed_label_text", ""),
                reason=record.get("reason", ""),
                verdict=claim_verdict(sample_id, record).get("verdict"))
        per_sample[sample_id] = entry

    payload = dict(summary=summary, fidelity=fidelity, flips=flips, audited=audited,
                   per_sample=per_sample, conditions_order=conditions_order,
                   audited_alignment=alignment,
                   audited_expected=AUDITED_EXPECTED,
                   chart_status=Counter(v["chart_status"] for v in chart_facts.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")

    lines = ["# Router 归因实验结果", ""]
    lines.append("## 0. 条件说明")
    lines.append("")
    lines.append("| 条件 | 选项 | 提示词 | 用途 |")
    lines.append("|---|---|---|---|")
    lines.append("| c0_options_production | 可见 | 生产原文 | 复现冻结版本，校验实验装置 |")
    lines.append("| c1_blind_production | 屏蔽 | 生产原文 | c0→c1 量化**选项污染** |")
    lines.append("| c2_blind_quote | 屏蔽 | 生产原文＋举证要求 | c1→c2 量化**举证要求**的效果 |")
    lines.append("| c3_blind_two_stage | 屏蔽 | 两问式重构 | c1→c3 量化**拆两阶段**的效果 |")
    lines.append("")
    lines.append("## 1. 装置保真度（c0 对冻结 v4）")
    lines.append("")
    lines.append(f"- 一致 {fidelity['matched']} 题；不一致 {len(fidelity['mismatched'])} 题；缺记录 {len(fidelity['missing'])} 题")
    for item in fidelity["mismatched"][:10]:
        lines.append(f"  - Q{item['sample_id']}: 冻结={item['frozen']} → 复跑={item['rerun']}")
    lines.append("")
    lines.append("## 2. 各条件的三分类分布")
    lines.append("")
    lines.append("| 条件 | printed_values | coordinate_reading | qualitative | 进 Geometry |")
    lines.append("|---|---:|---:|---:|---:|")
    for condition in conditions_order:
        s = summary[condition]
        src = s["evidence_source"]
        lines.append(f"| {condition} | {src.get('printed_values',0)} | {src.get('coordinate_reading',0)} "
                     f"| {src.get('qualitative',0)} | {s['needs_geometry'].get(True,0)} |")
    lines.append("")
    lines.append("## 3. printed_values 的可核实性")
    lines.append("")
    for condition in conditions_order:
        s = summary[condition]
        lines.append(f"**{condition}** — 共 {s['printed_values']} 条 printed_values")
        for verdict, count in sorted(s["verdicts"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {count:>3}  {verdict}")
        if s["unsupported_printed_claims"]:
            lines.append(f"  - **无法被图中文本支持的 printed_values：{len(s['unsupported_printed_claims'])} 题** "
                         f"（引用数字在图里不存在，或只存在于轴刻度）"
                         f"：{', '.join('Q'+x for x in s['unsupported_printed_claims'])}")
        lines.append("")
    lines.append("## 4. 相对 c0 的判断翻转")
    lines.append("")
    for condition in conditions_order:
        if condition not in flips:
            continue
        f = flips[condition]
        lines.append(f"**{condition}** — 翻转 {f['changed']} / {summary[condition]['n']} 题")
        for item in f["detail"][:20]:
            lines.append(f"  - Q{item['sample_id']}: {item['before']} → {item['after']}")
        lines.append("")
    lines.append("## 5. 与人审结论的分流对齐率")
    lines.append("")
    lines.append("期望分流取自项目自己的图像审计记录（`router_review.json`），不使用 Gold、不使用分数。")
    lines.append("")
    lines.append("| 条件 | 与审计结论一致 | 不一致的题 |")
    lines.append("|---|---:|---|")
    for condition in conditions_order:
        item = alignment[condition]
        bad = "、".join(f"Q{x['sample_id']}（期望 {x['expected']}，实得 {x['got']}）"
                        for x in item["mismatch"]) or "—"
        lines.append(f"| {condition} | {item['matched']} / {item['total']} | {bad} |")
    lines.append("")
    lines.append("## 6. 已确认误判案例在各条件下的表现")
    lines.append("")
    for sid, item in audited.items():
        lines.append(f"### Q{sid}")
        lines.append(f"- 审计结论：{item['note']}")
        lines.append(f"- 题目：{(item['question'] or '')[:150]}")
        chart = item.get("chart") or {}
        lines.append(f"- 已标定：{chart.get('calibrated')} · 非刻度数字：{chart.get('non_tick_numeric', [])[:12]}"
                     f" · 边缘疑似轴刻度：{chart.get('edge_like_numeric', [])[:12]}"
                     f" · 图内数字：{chart.get('in_plot_numeric', [])[:12]}")
        lines.append("")
        lines.append("| 条件 | 三分类 | 要求数值 | 目标有印刷标签 | 引用的标签原文 | 核实 |")
        lines.append("|---|---|---|---|---|---|")
        for condition, value in item["conditions"].items():
            lines.append(f"| {condition} | {value['source']} | {value['requires_numeric']} "
                         f"| {value['has_printed']} | {value['printed_label'] or '—'} | {value['verdict']} |")
        lines.append("")
        for condition, value in item["conditions"].items():
            if value.get("reason"):
                lines.append(f"- `{condition}` reason: {value['reason'][:200]}")
        lines.append("")

    report = args.output.with_name("ATTRIBUTION_REPORT.md")
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:80]))
    print()
    print("wrote", args.output)
    print("wrote", report)


if __name__ == "__main__":
    main()
