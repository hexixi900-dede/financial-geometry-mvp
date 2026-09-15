# Phase 9 当前版本

2026-09-15：question_router_v4已完成修改、通过相关检查并冻结，64题pilot已完成，Raw31→30；8完整/4部分/12无值。full未启动，不会自动启动。
具体运行状态以status.txt与queue_status.json为准。

- 代码：phase9_releases/20260915_question_router_v4
- 输出：phase9_evidence/question_router_v4
- 入口：src/run_question_router.py --cohort pilot
- 终态：pilot/outcomes.jsonl；分流：pilot/routers.jsonl；得分：pilot/summary.json
- 代码与输入已冻结并校验SHA256，启动后禁止修改。上版结果均保留。

本版只修彩色折线被遮挡后短片段过早丢弃：同色真实像素先聚合，再检查覆盖；不填造遮挡处像素，不改neutral通道。
CPU隔离重放确认Q1491恢复4/4目标，Q1272恢复约-12的谷值。不是准确率提升证据。
Router/Planner/最终回答提示保持v3；不新增模块或回退规则；64题成员与顺序相同，全阶段重新推理。

## 最近结果
- question_router_v3：64/64已完成，Raw31→30，修对1/改错2；40Raw、24进入；7完整/5部分/12无值。完整测量组无得分变化，唯一修对无Geometry值。full未启动。
- Q1491/Q1960两题官方计为改错，但原图支持增强选项更多，已另记标注/题干疑点。官方评分不改，题目不删，推理不读取Gold。详见v3/RESULTS.md。
- question_router_v2：64题31→28；18Raw/46进入，8完整/3部分/35无值；full未启动。
- question_router_v1：64题31→26；full未启动。
- visual_feedback_v4：完整11099题6170→6142，净-28。
- visual_feedback_v3：完整11099题6170→6180，净+10。
- visual_feedback_v2：用户授权停止，保留449/3699候选结果。

64题是多轮开发集，不能当作独立测试结果。旧3699只是结构候选，不能称Router输出。
完整评测仍要求11099题统一Single/Multi/Calculation/Overall。使用同一原始Raw PNG、既有单位、min_pixels200704/max_pixels802816；无caption，Gold仅离线计分；进入Geometry后不回填缓存Raw。
Q43 Router误把估计值当印刷标签、季度OCR把Q识别成0、语义计划及同色多系列仍有未解决限制。

最新审查：Router确有虚构数据标签和数值/类别需求混淆；不能将其输出当已核实标签。界面已改模型判断措辞并显示Gold/Raw/系统来源。11099条对原始Arrow题干/选项/Gold及Raw记录核对无错配。下一步专门修分流，未启动新实验。详见question_router_v4/ROUTER_AND_DISPLAY_AUDIT.md。
