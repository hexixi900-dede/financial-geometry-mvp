# 当前方案与代码入口

更新日期：2026-09-08。本文描述最新实现；历史 Phase 报告保留原结果，不代表新版效果。

## 想验证的 idea

图表有直接数值标注时保留 Raw VLM 回答；无标注时，让 VLM 看原图、题目和所有选项，决定要测哪个系列、哪些点或哪段曲线。程序负责像素定位、坐标轴映射和数值计算，最后把测量证据与原图一起交给 VLM 理解单位并输出答案。Python 不再用数值正则强行匹配选项。

单选、多选和计算题共用流程。多选需要覆盖全部选项所需的测量目标；多步算式引用测量目标编号。定位正确是重要前提，但测量误差、系列混淆和模型决策仍可能造成错误，不能预先承诺准确率不下降。

## 代码怎么读

1. [src/vlm_semantic_planner.py](src/vlm_semantic_planner.py)：前端 Qwen2.5-VL 看原图及全部选项，输出目标类别、系列、归一化区域、首末点/点值/柱高/折线区间等结构化计划，不输出猜测的图表数值。
2. [src/phase9_plan_contract.py](src/phase9_plan_contract.py)：共享目标协议与确定性运算，含差值、增长率、比值、求和、均值、极值以及嵌套算式。保留目标编号；不把像素采样均值冒充时间平均值。
3. [src/phase8_geometry_adapter.py](src/phase8_geometry_adapter.py)：OCR 轴锚点、系列选择和像素测量。VLM 区域约束实际折线像素；浮动柱测上下端；有图例时区分指定柱段/系列。
4. [src/phase8_hybrid_qa.py](src/phase8_hybrid_qa.py)：准备候选集与 CPU 几何执行。当前统一流程使用 `--evidence-only`；无此标志的旧选项匹配路径仅保留历史兼容。
5. [src/vlm_evidence_answerer.py](src/vlm_evidence_answerer.py)：增强 Prompt、最终答案解析与评测。明确传入单选/多选/计算题型，后端仍然接收原图；测量或回答不可用时系统保留 Raw VLM。
6. [src/run_unified_resident.py](src/run_unified_resident.py)：模型只加载一次，按批次完成前端、CPU 测量和后端，再继续全量。逐题保存，支持断点续跑。
7. [apps/geometry_inspector.py](apps/geometry_inspector.py)：查看原图、VLM 定位框、标定线、测量值、算式、前后端实际请求/回复，以及 Raw 回退状态。历史 Python 答案单独展示。

## 实验状态

- 模型：本地 Qwen2.5-VL-7B-Instruct。
- 新版小批：65 道（5 道已讨论案例，加固定随机种子抽取的每题型 20 道）。用于排查，不能视为无偏全量估计。
- 新版候选集：3,699 道，其中单选 2,248、多选 826、数值计算 625；来自现有可标定单 y 轴、无数值标注的柱/折线候选筛选，不等于支持整个数据集所有图表。
- 正式 Raw 缓存对应全量 11,099 道：单选 4,465/6,567（68.0%），多选 1,065/2,680（39.7%），计算 640/1,852（34.6%）。评测使用 `qwen25vl7b_direct_full_per_sample.jsonl`，不是通用名称的结果文件。
- **截至本次更新，新版还在等待服务器 GPU，没有新版准确率或真实前后端回复可报告。** 已通过针对性的协议、几何和断点续跑检查；测试中的模拟回复只用于临时测试，不是实验结果。

## 运行与依赖

服务器启动入口：`bash scripts/run_unified.sh`。每 60 秒检查 GPU，检测到一次空闲后启动；小批后自动继续全量。模型在 CPU 测量阶段保持加载；遇到其他 GPU 进程时按现有调度逻辑退出让出并等待恢复。

运行入口目前沿用服务器的模型、OCR Python 和数据路径。`src/run_unified_resident.py --help` 提供模型、OCR 环境、基线文件及输出根目录参数；换机器需要设置这些路径并准备数据。查看界面需要安装 `requirements-inspector.txt`，然后运行 `streamlit run apps/geometry_inspector.py`。

外部 FinMME 数据、模型权重、OCR 缓存、逐题实验日志和密钥不在仓库。最新运行文件位于服务器 `phase9_evidence/unified/{pilot,full}/`，其中输入、计划、测量、Prompt、回复各自保存为 JSONL，完成后生成 `summary.json`。原始标注只用于事后评测，不传入前后端 Prompt。

## 接下来需要讨论什么

优先看实际测量链是否忠实执行 VLM 的目标计划，以及多选与计算题是否获得有效数值证据。等真实推理完成后，再区分错误发生在语义定位、轴标定/像素测量、算术还是最终单位/选项理解。当前目标是简明可验证的几何增强方法，不做逐题特殊规则，也不宣称支持所有图形。
