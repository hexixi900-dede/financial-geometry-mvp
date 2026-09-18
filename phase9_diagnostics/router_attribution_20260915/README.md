# Router 归因实验（question_router_v4)

独立一次性诊断。**不修改任何冻结版本，不写入 `phase9_evidence` 或 `phase9_releases`，不改变官方评分。**

## 要回答的问题

question_router_v4 在 64 题开发集上 Raw 31 → 系统 30（−1），且进入 Geometry 的 24 题里只有 8 题测出值。
Router 的题目级判断是否可靠？如果不可靠，错在哪一类原因上？

## 条件设计（逐项消融）

| 条件 | 选项 | 提示词 | 隔离出的变量 |
|---|---|---|---|
| `c0_options_production` | 可见 | 生产原文 | 复现冻结版本，校验实验装置 |
| `c1_blind_production` | 屏蔽 | 生产原文 | c0→c1 = **选项污染** |
| `c2_blind_quote` | 屏蔽 | 生产原文＋举证要求 | c1→c2 = **举证要求**的效果 |
| `c3_blind_two_stage` | 屏蔽 | 两问式拆分 | c1→c3 = **拆两阶段**的效果 |

同一原图、同一 `min_pixels=200704 / max_pixels=802816`、`do_sample=False`，不使用 Gold，不做人工逐题覆写。

`c3` 的两问：
1. 回答这道题**需不需要一个数值**？（"哪一年/哪个季度最高"只需定位并命名类别，属于视觉比较；"最高值是多少/相差多少"才需要数值）
2. 只有第 1 问为是时：需要的数字是否以**印刷标签**形式出现在图中对应标记上？（轴刻度、题干里的数字、模型自己的估计都不算）

## 核实方式（不依赖 Gold）

先用项目**自己的** CPU 工具链 `geometry_toolbox.prepare_chart` 对 64 张图做一次 OCR，
记录全图文本、被轴标定占用的 token（即轴刻度）、以及每个 token 是否落在绘图区内。
然后把每条 `printed_values` 声称引用的数字与这份清单比对：

- `absent_from_image` — 引用的数字整张图里都不存在 → 虚构
- `only_axis_tick` — 引用的数字只是轴刻度 → 轴刻度混淆
- `likely_uncalibrated_axis_tick` — 紧贴绘图区边缘且与其他数字同列 → 疑似未标定的另一侧轴
- `number_exists_in_chart_text` / `located_in_plot` — 图中确有 → 可能是真标签
- `number_present_in_options` — 该数字同时出现在选项里 → 选项污染线索

## 结果

### 装置保真度
`c0` 与冻结 v4 的三分类 **64/64 完全一致** → 实验装置可信。

### 三分类分布（64 题）

| 条件 | printed_values | coordinate_reading | qualitative | 进入 Geometry |
|---|---:|---:|---:|---:|
| c0 基线 | 26 | 24 | 14 | 24 |
| c1 屏蔽选项 | 23 | 30 | 11 | 30 |
| c2 ＋举证要求 | 18 | 37 | 9 | 37 |
| c3 两问式拆分 | 16 | 26 | 22 | 26 |

### 与人审结论的分流对齐率

期望分流取自项目自己的图像审计记录 `phase9_evidence/question_router_v4/router_review.json`。

| 条件 | 与审计一致 |
|---|---:|
| c0 基线 | **0 / 6** |
| c1 屏蔽选项 | 0 / 6 |
| c2 ＋举证要求 | 1 / 6 |
| c3 两问式拆分 | **5 / 6** |

唯一残留的 Q828 是"答案类型"判断问题：题干出现 "highest value"，模型据此认为需要数值，
但选项全是年份，答案其实是类别 → 第 1 问的规则必须按**答案是什么**来判，而不是按题干里有没有 "value" 这个词。

### 三个关键结论

1. **选项污染是真的，而且是最主要的一类错。**
   Q43 图上只有轴刻度、一个数据标签都没有，c0 却说 "clearly labeled as approximately 70 MYR bn" ——
   70 恰好是选项 B。Q18 同型（30 恰好是选项 C）。

2. **让模型举证是无效的。**
   c2 要求逐字抄写标签原文并指明它附着在哪个标记上。模型照做了，但仍在编：
   Q43 回答 `"70" is printed directly on the bar`，而全图根本不存在 70；
   Q11095 抄出 `3.0`，那是轴刻度。**自述不可信，必须外部核验。**

3. **真正的病因是"类别定位"与"数值恢复"混为一谈，而拆成两问就能修掉。**
   Q1960（问哪个季度最高）、Q1491（问哪些季度最接近最低）在 c0/c1/c2 三组里全部被判 `coordinate_reading`
   送进几何，答成 B / A，而 Raw 本来答对 C / D —— 这就是那 2 处改错的来源。
   c3 把这两题判为 `qualitative`（不需要数值恢复）。

## 复现

```bash
REPO=/data/liu_jun/financial_geometry_mvp
D=$REPO/phase9_diagnostics/router_attribution_20260915
IN=$REPO/phase9_evidence/question_router_v4/pilot/geometry_inputs.jsonl
OCR_PY=/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python
GPU_PY=/data/liu_jun/finmme_reproduction/.venv_phase2a/bin/python

# 1) 图表文本清单（CPU／OCR，约 6 分钟）
CUDA_VISIBLE_DEVICES='' $OCR_PY -u $D/build_text_inventory.py --repo $REPO --inputs $IN \
  --output $D/text_inventory.jsonl --ocr-models \
  /data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models

# 2) 四条件复跑 Router（GPU，约 30 分钟，256 次调用）
$GPU_PY -u $D/run_router_conditions.py --repo $REPO --inputs $IN --output $D/router_conditions.jsonl

# 3) 归因分析
python3 $D/analyze_attribution.py --diagnostics $D --inventory $D/text_inventory.jsonl \
  --conditions $D/router_conditions.jsonl \
  --frozen-routers $REPO/phase9_evidence/question_router_v4/pilot/routers.jsonl \
  --review $REPO/phase9_evidence/question_router_v4/router_review.json \
  --inputs $IN --output $D/attribution.json
```

两个脚本都支持续跑（按 `sample_id`／`(sample_id, condition)` 跳过已完成记录）。

## 产物

- `text_inventory.jsonl` — 64 题图表文本清单
- `router_conditions.jsonl` — 256 条四条件判断记录
- `attribution.json` — 机器可读结果（含 `per_sample`，供审查页读取）
- `ATTRIBUTION_REPORT.md` — 完整报告

## 局限

- 期望分流只有 6 题有人审依据。其余 58 题本轮**不产出**分流真值，因此不能据本实验宣称准确率提升。
- 轴刻度的识别依赖项目自身的标定；当某一侧轴拟合失败时，该侧刻度会落在绘图区内，
  用"紧贴边缘且与其他数字同列"启发式补判，仍可能漏判。
- 64 题是反复使用的开发集，不是独立测试集。
- 本实验只诊断 Router，不涉及 Planner、测量与最终回答。
