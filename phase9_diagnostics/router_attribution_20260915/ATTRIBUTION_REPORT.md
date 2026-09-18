# Router 归因实验结果

## 0. 条件说明

| 条件 | 选项 | 提示词 | 用途 |
|---|---|---|---|
| c0_options_production | 可见 | 生产原文 | 复现冻结版本，校验实验装置 |
| c1_blind_production | 屏蔽 | 生产原文 | c0→c1 量化**选项污染** |
| c2_blind_quote | 屏蔽 | 生产原文＋举证要求 | c1→c2 量化**举证要求**的效果 |
| c3_blind_two_stage | 屏蔽 | 两问式重构 | c1→c3 量化**拆两阶段**的效果 |

## 1. 装置保真度（c0 对冻结 v4）

- 一致 64 题；不一致 0 题；缺记录 0 题

## 2. 各条件的三分类分布

| 条件 | printed_values | coordinate_reading | qualitative | 进 Geometry |
|---|---:|---:|---:|---:|
| c0_options_production | 26 | 24 | 14 | 24 |
| c1_blind_production | 23 | 30 | 11 | 30 |
| c2_blind_quote | 18 | 37 | 9 | 37 |
| c3_blind_two_stage | 16 | 26 | 22 | 26 |

## 3. printed_values 的可核实性

**c0_options_production** — 共 26 条 printed_values
  -   8  no_number_claimed
  -   2  number_exists_in_chart_text|number_present_in_question
  -   2  only_axis_tick|number_present_in_options|number_present_in_question
  -   2  number_exists_in_chart_text|not_in_plot|number_present_in_options|number_present_in_question
  -   2  only_axis_tick
  -   2  number_exists_in_chart_text|located_in_plot|number_present_in_question
  -   1  number_exists_in_chart_text
  -   1  absent_from_image|number_present_in_options|number_present_in_question
  -   1  likely_uncalibrated_axis_tick
  -   1  only_axis_tick|number_present_in_question
  -   1  number_exists_in_chart_text|not_in_plot|number_present_in_options
  -   1  number_exists_in_chart_text|located_in_plot|number_present_in_options
  -   1  number_exists_in_chart_text|number_present_in_options|number_present_in_question
  -   1  number_exists_in_chart_text|not_in_plot|number_present_in_question
  - **无法被图中文本支持的 printed_values：7 题** （引用数字在图里不存在，或只存在于轴刻度）：Q11095, Q18, Q1810, Q2500, Q42, Q43, Q828

**c1_blind_production** — 共 23 条 printed_values
  -   5  no_number_claimed
  -   3  number_exists_in_chart_text|not_in_plot|number_present_in_options|number_present_in_question
  -   3  only_axis_tick
  -   3  number_exists_in_chart_text|not_in_plot|number_present_in_question
  -   2  number_exists_in_chart_text|number_present_in_question
  -   1  number_exists_in_chart_text
  -   1  absent_from_image|number_present_in_options|number_present_in_question
  -   1  only_axis_tick|number_present_in_options|number_present_in_question
  -   1  number_exists_in_chart_text|located_in_plot|number_present_in_question
  -   1  absent_from_image
  -   1  number_exists_in_chart_text|located_in_plot
  -   1  number_exists_in_chart_text|number_present_in_options|number_present_in_question
  - **无法被图中文本支持的 printed_values：6 题** （引用数字在图里不存在，或只存在于轴刻度）：Q11, Q11095, Q18, Q20, Q42, Q43

**c2_blind_quote** — 共 18 条 printed_values
  -   6  number_exists_in_chart_text
  -   5  absent_from_image
  -   3  number_exists_in_chart_text|located_in_plot
  -   2  only_axis_tick
  -   1  absent_from_image|number_present_in_options
  -   1  number_exists_in_chart_text|not_in_plot
  - **无法被图中文本支持的 printed_values：8 题** （引用数字在图里不存在，或只存在于轴刻度）：Q11, Q11095, Q1320, Q20, Q2455, Q43, Q4599, Q77

**c3_blind_two_stage** — 共 16 条 printed_values
  -   6  no_number_claimed
  -   3  only_axis_tick
  -   3  absent_from_image
  -   2  number_exists_in_chart_text
  -   2  number_exists_in_chart_text|located_in_plot
  - **无法被图中文本支持的 printed_values：6 题** （引用数字在图里不存在，或只存在于轴刻度）：Q1272, Q1320, Q2455, Q42, Q4599, Q77

## 4. 相对 c0 的判断翻转

**c1_blind_production** — 翻转 8 / 64 题
  - Q828: printed_values → coordinate_reading
  - Q1810: printed_values → coordinate_reading
  - Q2333: printed_values → coordinate_reading
  - Q3232: qualitative → coordinate_reading
  - Q4128: qualitative → coordinate_reading
  - Q5390: coordinate_reading → qualitative
  - Q0: qualitative → coordinate_reading
  - Q13: qualitative → coordinate_reading

**c2_blind_quote** — 翻转 18 / 64 题
  - Q18: printed_values → coordinate_reading
  - Q124: printed_values → coordinate_reading
  - Q828: printed_values → coordinate_reading
  - Q1320: coordinate_reading → printed_values
  - Q1777: printed_values → coordinate_reading
  - Q1810: printed_values → coordinate_reading
  - Q2333: printed_values → coordinate_reading
  - Q2500: printed_values → coordinate_reading
  - Q2577: qualitative → coordinate_reading
  - Q2636: qualitative → coordinate_reading
  - Q3232: qualitative → coordinate_reading
  - Q4128: qualitative → coordinate_reading
  - Q5390: coordinate_reading → qualitative
  - Q5422: printed_values → qualitative
  - Q0: qualitative → coordinate_reading
  - Q13: qualitative → coordinate_reading
  - Q24: qualitative → coordinate_reading
  - Q42: printed_values → coordinate_reading

**c3_blind_two_stage** — 翻转 24 / 64 题
  - Q43: printed_values → coordinate_reading
  - Q1960: coordinate_reading → qualitative
  - Q18: printed_values → coordinate_reading
  - Q124: printed_values → coordinate_reading
  - Q11095: printed_values → coordinate_reading
  - Q828: printed_values → coordinate_reading
  - Q869: coordinate_reading → qualitative
  - Q1272: coordinate_reading → printed_values
  - Q1320: coordinate_reading → printed_values
  - Q1491: coordinate_reading → qualitative
  - Q1777: printed_values → qualitative
  - Q1810: printed_values → coordinate_reading
  - Q2333: printed_values → qualitative
  - Q2500: printed_values → coordinate_reading
  - Q5390: coordinate_reading → qualitative
  - Q5422: printed_values → qualitative
  - Q1: qualitative → coordinate_reading
  - Q6: printed_values → coordinate_reading
  - Q11: printed_values → coordinate_reading
  - Q5: printed_values → qualitative

## 5. 与人审结论的分流对齐率

期望分流取自项目自己的图像审计记录（`router_review.json`），不使用 Gold、不使用分数。

| 条件 | 与审计结论一致 | 不一致的题 |
|---|---:|---|
| c0_options_production | 0 / 6 | Q43（期望 coordinate_reading，实得 printed_values）、Q18（期望 coordinate_reading，实得 printed_values）、Q11095（期望 coordinate_reading，实得 printed_values）、Q828（期望 qualitative，实得 printed_values）、Q1960（期望 qualitative，实得 coordinate_reading）、Q1491（期望 qualitative，实得 coordinate_reading） |
| c1_blind_production | 0 / 6 | Q43（期望 coordinate_reading，实得 printed_values）、Q18（期望 coordinate_reading，实得 printed_values）、Q11095（期望 coordinate_reading，实得 printed_values）、Q828（期望 qualitative，实得 coordinate_reading）、Q1960（期望 qualitative，实得 coordinate_reading）、Q1491（期望 qualitative，实得 coordinate_reading） |
| c2_blind_quote | 1 / 6 | Q43（期望 coordinate_reading，实得 printed_values）、Q11095（期望 coordinate_reading，实得 printed_values）、Q828（期望 qualitative，实得 coordinate_reading）、Q1960（期望 qualitative，实得 coordinate_reading）、Q1491（期望 qualitative，实得 coordinate_reading） |
| c3_blind_two_stage | 5 / 6 | Q828（期望 qualitative，实得 coordinate_reading） |

## 6. 已确认误判案例在各条件下的表现

### Q43
- 审计结论：Router声称“70 MYR bn已标注”，原图只有轴刻度和折线，目标点没有70标签。应读取数值，属于分流误判。
- 题目：What was the total loan value applied in January 2021?
- 已标定：True · 非刻度数字：['3', '9.39885e+17'] · 边缘疑似轴刻度：[] · 图内数字：[]

| 条件 | 三分类 | 要求数值 | 目标有印刷标签 | 引用的标签原文 | 核实 |
|---|---|---|---|---|---|
| c0_options_production | printed_values | None | None | — | absent_from_image|number_present_in_options|number_present_in_question |
| c1_blind_production | printed_values | None | None | — | absent_from_image|number_present_in_options|number_present_in_question |
| c2_blind_quote | printed_values | None | None | 70 | absent_from_image|number_present_in_options |
| c3_blind_two_stage | coordinate_reading | True | False | — | no_number_claimed |

- `c0_options_production` reason: The total loan value applied in January 2021 is clearly labeled as approximately 70 MYR bn.
- `c1_blind_production` reason: The total loan value applied in January 2021 is directly labeled on the graph as 70 MYR bn.
- `c2_blind_quote` reason: The value '70' is printed directly on the bar corresponding to January 2021.
- `c3_blind_two_stage` reason: brief

### Q18
- 审计结论：Router把约30的轴读数称作直接标签。目标点没有数值标注；问题问P/E数值，应进入几何读取。
- 题目：What was the P/E ratio of Sinobiopharma at the beginning of January 2021?
- 已标定：True · 非刻度数字：['-21', '-22', '-23', '-24'] · 边缘疑似轴刻度：[] · 图内数字：['-21', '-22', '-23', '-24']

| 条件 | 三分类 | 要求数值 | 目标有印刷标签 | 引用的标签原文 | 核实 |
|---|---|---|---|---|---|
| c0_options_production | printed_values | None | None | — | only_axis_tick|number_present_in_options|number_present_in_question |
| c1_blind_production | printed_values | None | None | — | only_axis_tick|number_present_in_options|number_present_in_question |
| c2_blind_quote | coordinate_reading | None | None | — | absent_from_image|number_present_in_question |
| c3_blind_two_stage | coordinate_reading | True | False | — | no_number_claimed |

- `c0_options_production` reason: The P/E ratio of Sinobiopharma at the beginning of January 2021 is clearly labeled as 30.
- `c1_blind_production` reason: The P/E ratio of Sinobiopharma at the beginning of January 2021 is directly labeled as 30 on the graph.
- `c2_blind_quote` reason: The P/E ratio is not directly printed on the graph. The x-axis shows months, and the y-axis shows values, but there is no direct label indicating the P/E ratio. To find the P/E ratio, we need to read 
- `c3_blind_two_stage` reason: brief

### Q11095
- 审计结论：Router声称Saudi/Kuwait有3.0和4.5标签，原图没有这些目标标签。它还把浮动柱的端点/高度混淆；本题数值差需要测量。
- 题目：How much more spare capacity does Saudi have compared to Kuwait in mb/d?
- 已标定：True · 非刻度数字：['20', '8'] · 边缘疑似轴刻度：[] · 图内数字：[]

| 条件 | 三分类 | 要求数值 | 目标有印刷标签 | 引用的标签原文 | 核实 |
|---|---|---|---|---|---|
| c0_options_production | printed_values | None | None | — | only_axis_tick |
| c1_blind_production | printed_values | None | None | — | only_axis_tick |
| c2_blind_quote | printed_values | None | None | 3.0 | only_axis_tick |
| c3_blind_two_stage | coordinate_reading | True | False | — | no_number_claimed |

- `c0_options_production` reason: The spare capacity for Saudi Arabia and Kuwait is directly labeled on the bar graph as 3.0 mb/d and 4.5 mb/d respectively.
- `c1_blind_production` reason: The spare capacity for Saudi Arabia and Kuwait is directly labeled on the bar graph as 3.0 mb/d and 4.5 mb/d respectively.
- `c2_blind_quote` reason: The spare capacity for Saudi Arabia is clearly labeled as 3.0 mb/d.
- `c3_blind_two_stage` reason: brief

### Q828
- 审计结论：Router引用右轴10%作为数据标签，类别错误。题目问最高值对应哪一年，属于可先视觉比较的类别问题，不必因无数值标签就做数值恢复。
- 题目：Which of the following years shows the highest value for the trailing margin percentage?
- 已标定：True · 非刻度数字：['10', '12', '5', '6', '8.88889e+18'] · 边缘疑似轴刻度：['10', '5'] · 图内数字：['10', '12', '5', '6']

| 条件 | 三分类 | 要求数值 | 目标有印刷标签 | 引用的标签原文 | 核实 |
|---|---|---|---|---|---|
| c0_options_production | printed_values | None | None | — | likely_uncalibrated_axis_tick |
| c1_blind_production | coordinate_reading | None | None | — | likely_uncalibrated_axis_tick |
| c2_blind_quote | coordinate_reading | None | None | — | no_number_claimed |
| c3_blind_two_stage | coordinate_reading | True | False | — | no_number_claimed |

- `c0_options_production` reason: The highest value for the trailing margin percentage is clearly labeled as 10% on the right y-axis.
- `c1_blind_production` reason: The highest value for the trailing margin percentage (orange line) appears to be around 10%, which is not directly labeled on the graph. It must be estimated by reading the height of the orange line a
- `c2_blind_quote` reason: The highest value for the trailing margin percentage is not directly labeled on the graph, requiring an estimate based on the plotted position.
- `c3_blind_two_stage` reason: brief

### Q1960
- 审计结论：题目问最高点的季度，不直接要求最高数值；Router将类别定位误说成读取y值。另：Gold C来自原数据；图中B约27.5高于C约26，全局最高不在选项，存在题干/标注疑点。
- 题目：In which quarter did the Non-Interest Income/Operating Income percentage reach its highest point in the provided chart?
- 已标定：True · 非刻度数字：['2.82888e+17', '7'] · 边缘疑似轴刻度：[] · 图内数字：['7']

| 条件 | 三分类 | 要求数值 | 目标有印刷标签 | 引用的标签原文 | 核实 |
|---|---|---|---|---|---|
| c0_options_production | coordinate_reading | None | None | — | no_number_claimed |
| c1_blind_production | coordinate_reading | None | None | — | no_number_claimed |
| c2_blind_quote | coordinate_reading | None | None | — | no_number_claimed |
| c3_blind_two_stage | qualitative | False | None | — | no_number_claimed |

- `c0_options_production` reason: The highest point on the graph is not labeled with a specific quarter, so it must be estimated by reading the graph's scale.
- `c1_blind_production` reason: The highest point on the graph is not labeled with a specific value, requiring an estimate based on the plotted position.
- `c2_blind_quote` reason: The highest point on the graph is not labeled with a specific value, requiring an estimate based on the plotted position.

### Q1491
- 审计结论：题目问最接近最低值的季度，可以先视觉比较。Gold D确为原数据标注；原图选项A更接近最低，另记标注疑点，评分未改。
- 题目：Which quarters are closest to the lowest DOI value in the chart?
- 已标定：True · 非刻度数字：['2.82829e+20'] · 边缘疑似轴刻度：[] · 图内数字：[]

| 条件 | 三分类 | 要求数值 | 目标有印刷标签 | 引用的标签原文 | 核实 |
|---|---|---|---|---|---|
| c0_options_production | coordinate_reading | None | None | — | no_number_claimed |
| c1_blind_production | coordinate_reading | None | None | — | no_number_claimed |
| c2_blind_quote | coordinate_reading | None | None | — | only_axis_tick|number_present_in_options |
| c3_blind_two_stage | qualitative | False | None | — | no_number_claimed |

- `c0_options_production` reason: The lowest DOI value in the chart is not directly labeled with a quarter. It must be estimated by reading the position on the y-axis.
- `c1_blind_production` reason: The lowest DOI value in the chart is not directly labeled with a specific quarter. It must be estimated by reading the y-axis value corresponding to the lowest point on the red line.
- `c2_blind_quote` reason: The lowest DOI value in the chart appears to be around the 80 mark on the y-axis, which corresponds to the 1Q14 quarter.

