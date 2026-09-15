# 当前进度（2026-09-15）

当前代码对应服务器已冻结的 **question_router_v4**，运行入口为 [src/run_question_router.py](src/run_question_router.py)。当前可确认的结论是工具漏段修复有效，**尚未证明题目级Router框架提高问答准确率**。

## 方法
原图＋问题＋选项 → 独立判断是否需要几何数值 → 不需要则Raw；需要则Target Planner → Axis/Legend/Bar/Line工具 → Python运算 → VLM结合原图、测量证据回答。
需要Geometry与工具能否支持分别记录。进入后不回填缓存Raw；最多允许一次看定位图后修正计划。逐期平均必须测实际各期，不能平均像素采样序列。
使用Raw同一原始PNG、已有单位及min_pixels=200704/max_pixels=802816，无额外caption、无Gold进入推理。

## 已完成结果
|版本|范围|Raw正确|系统正确|
|---|---|---:|---:|
|visual_feedback_v3|全量11,099|6170|6180|
|visual_feedback_v4|全量11,099|6170|6142|
|question_router_v1|开发64|31|26|
|question_router_v2|开发64|31|28|
|question_router_v3|开发64|31|30|
|question_router_v4|开发64|31|30|

前两项仍使用历史结构候选流程，不能作为当前独立Router的全量结果。旧3,699是结构预筛候选，不是Router输出。
当前v4：40题Raw、24题进入；8完整测量、4部分、12无值；修对1/官方计改错2。唯一修对没有成功Geometry数值，不能归因为几何贡献。64题反复用于开发，不是独立测试集。**本版全量未启动。**
汇总见 [results/phase9_progress_20260915](results/phase9_progress_20260915)。

## 最新修复和未解决问题
- 全日期锚点、季度OCR三位年份误解析、黑色细线图例/文字污染已修复。
- 同色折线被参考线遮挡后，先聚合真实片段再检查覆盖，保留缺口；Q1491恢复全部4个目标，Q1272恢复可见谷值。
- **Router尚不可靠**：Q43/Q18/Q11095虚构数值标签，Q828将轴刻度误当数据标签；类别/视觉比较与数值恢复需求也有混淆。下一项优先修Router，不继续把其判断当可靠前提跑全量。
- 对照原始FinMME Arrow、Raw预测与界面，11,099题题干/选项/Gold无错配；Raw记录一致，64题图片SHA及最终显示与评分导出一致。
- Q1491/Q1960有图像与Gold的标注/题干疑点，已单独记录，官方Gold和所有题目保留，不据Gold反向改推理。
- 界面明确显示“Router模型判断”，Gold/Raw/系统答案及来源对照。分组/堆叠/双轴/混合工具已有实现与局部检查，不能据此宣称全面可靠。

## 代码和复现入口
- [src/vlm_need_router.py](src/vlm_need_router.py)：独立判断；当前已知缺陷见上文。
- [src/framework_plan.py](src/framework_plan.py)、[src/geometry_toolbox.py](src/geometry_toolbox.py)：目标协议、工具组合。
- [src/framework_answer.py](src/framework_answer.py)：测量证据、可选计划修正和最终回答。
- [src/evaluate_question_router.py](src/evaluate_question_router.py)：全量与开发集分开计分。
- [apps/geometry_inspector.py](apps/geometry_inspector.py)：事后白盒审查，不调用模型。
- [scripts/prepare_question_router.py](scripts/prepare_question_router.py)：全量/开发输入；[scripts/activate_question_router.py](scripts/activate_question_router.py)：冻结代码与输入后排队。
- 入口支持 --root、--cohort pilot/full；默认pilot，full不会自动启动。旧run_unified.sh/run_phase9.sh不是当前入口。
- 启动脚本仍含实验服务器路径，迁移需配置模型、OCR、Raw与数据位置；并非开箱即用的跨机器发行包。
- 本仓库不含数据集、模型、完整逐题日志及运行缓存。发布的server_frozen_release_manifest.json记录服务器冻结来源；其输入文件不随仓库发布。

本次像素修复通过2项新增回归、29项相关检查、2项分流/评分流程检查；这些不等于准确率提升。最新变更没有重新跑旧阶段实验。
