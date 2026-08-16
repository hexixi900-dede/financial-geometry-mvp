# Calibrated Geometry Reasoning for Financial Charts — MVP Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-16T01:40:17+08:00
- Verification Status: ANALYZED
- Version Label: financial_geometry_mvp_v1

## 结论

- **Geometry recovery：可用于受限 MVP。** 测试集 Raw Geometry 的 MAE 为 `1.76027`，中位绝对误差为 `0.0743323`；平均/中位 axis-normalized error 分别为 `1.39%` / `0.40%`，`95.45%` 的样本落在 y 轴跨度 5% 以内。Bar 测试证据较强；Line 测试仅 `4` 张，应视为探索性结果。
- **Calibration：有明确额外价值（raw-unit 与 axis-normalized 配对 bootstrap 95% CI 均支持改善）。** 测试集 MAE `1.76027 → 1.42157`，平均 axis-normalized error `1.39% → 1.24%`。配对 chart bootstrap 的 `Raw MAE - Calibrated MAE` 为 `0.338698`，95% CI `[0.0162004, 0.867523]`；axis-normalized 差值为 `0.16%`，95% CI `[0.01%, 0.29%]`。增益主要由 Bar 支持，Line 尚不足以单独确认 Calibration 价值。
- 两个判断彼此独立；即使 calibrator 没有稳定增益，也不能据此否定像素到数值的 Geometry 路线。

## MVP 范围

本轮仅处理单 y 轴、简单垂直 Bar 与 Line 的 masked-value 恢复；未实现完整 QA、RL、Agent、LoRA、多轴、Pie、Area、Boxplot，也未调用本地 VLM。Geometry 由论文规格的最小重写组成：axis localization、pixel-to-value interpolation、bar detection/bar-top recovery、line edge-point recovery。

## FinMME 数据审计

- 复用现有离线数据：`/data/liu_jun/finmme_reproduction/data/FinMME_train`，没有重复下载。
- 复用现有图像缓存：`/data/liu_jun/finmme_reproduction/outputs/cache/phase2c_official_jpeg/`。
- 全量 `11,099` 题；原始 Arrow 审计为 4,450 张唯一图，现有官方预处理 JPEG 按 SHA-256 去重后为 `4,446` 张（4 组在缩放/JPEG 后发生内容合并）。
- 题型：Single `6,567`、Multi `2,680`、Numerical/Calculation `1,852`。
- `answer` 是 FinMME Gold；Numerical 同时提供 `unit` 与绝对 `tolerance`。1,852 条 Numerical 中 1 条 tolerance 为 NaN，其余可用。
- `1,649` 张图至少关联一道 Numerical 题；这些字段只用于数据审计与结果追踪，不作为 masked label 的 Geometry 输入。

官方来源：[FinMME GitHub](https://github.com/luo-junyu/FinMME)、[FinMME dataset](https://huggingface.co/datasets/luojunyu/FinMME)。

## ChartAgent 方法适配

[ChartAgent ACL 2026](https://aclanthology.org/2026.acl-long.843/) 与 [arXiv 2510.04514](https://arxiv.org/abs/2510.04514) 的附录公开了 Python 风格工具签名、参数、返回值与示例，但未找到作者发布的可执行官方仓库。因此本项目没有复现完整 Agent，而是按附录规格独立重写以下最小模块：

1. `axis_localizer`：EasyOCR 检测左 y 轴数值 tick，以 RANSAC 式配对和线性拟合返回 `(value, pixel_y)`；右 y 轴同时检查常规与反向刻度，拒绝右轴、额外 y 轴和多面板证据。
2. `interpolate_pixel_to_value`：由拟合的 `value = slope × pixel_y + intercept` 映射目标像素。
3. `get_bar` / bar-top recovery：颜色量化、连通域、矩形/基线/堆叠检查后返回目标柱 bbox 与 top y。
4. `get_edgepoint`：对彩色折线局部列扫描和加权线性拟合，恢复目标 `(x, y)`；存在多个 bar 组件时拒绝组合图。

## Masked-value 构造与 Gold 隔离

- EasyOCR 在原图中找数值文本；候选必须位于单 y 轴绘图区、落在显示范围内、不是 y tick，也不是三个以上水平对齐的 x-axis tick。
- 只接受全图 OCR 与独立放大裁剪 OCR 数值完全一致的标签；旋转标签会额外检查 90°/270°。
- Bar 候选若存在 3 条及以上斜线段证据，按 Bar+Line 组合图拒绝；同一 x 范围内若有上下相接的填充组件，按 stacked Bar 拒绝。Line 候选若绘图区颜色填充占比超过 12%，按“柱顶色带误识别为线”拒绝。三条规则不读取 Gold 或恢复误差。
- 遮挡使用 OCR 文本四边形和 OpenCV inpainting；若文本区域接触柱顶/折线点则拒绝。年份、坐标轴、图例和其他文本不遮。
- **Gold 防泄漏**：mask 后 Geometry 只接收空文本、`numeric_value=None` 的 target locator；pseudo-Gold 直到 raw value 生成后才用于误差计算。结果中的 `geometry_gold_access=False` 对每条样本显式保存。
- 筛选不读取 `abs(raw - Gold)`，因此没有用恢复误差挑“好看样本”。

## 样本流与实际数量

| 阶段 | 数量 |
|---|---:|
| FinMME 预处理唯一图 | 4,446 |
| 至少有 Numerical 的图 | 1,649 |
| 已执行 OCR/结构筛选 | 929 |
| 结构候选 viable | 107 |
| mask 后最终样本 | 87 |
| Bar / Line | 72 / 15 |

主要拒绝原因：无稳定线性左 y 轴 `181`，多轴/多面板 `104`，没有能与柱/点建立结构联系的数据标签 `537`；mask/crop/严格简单图范围二次审查失败 `20`。若最终少于约 100 条，应把它解释为 FinMME 在严格单轴、显式标签、可无损遮挡条件下的实际供给限制，而不是强行补齐。

## Geometry 结果

| Split | Method | N | MAE | Median AE | RMSE | Mean axis-N error | Median axis-N error | ≤1% axis | ≤5% axis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test | Raw | 22 | 1.76027 | 0.0743323 | 6.43404 | 1.39% | 0.40% | 90.91% | 95.45% |
| Test | Calibrated | 22 | 1.42157 | 0.0716219 | 6.15821 | 1.24% | 0.26% | 90.91% | 95.45% |
| All | Raw | 87 | 0.96997 | 0.125997 | 3.64828 | 0.97% | 0.45% | 88.51% | 96.55% |
| All | Calibrated | 87 | 0.717732 | 0.0672455 | 3.27073 | 0.78% | 0.22% | 87.36% | 96.55% |

原始单位跨百分比、金额与指数，MAE 用于同一样本上的 Raw/Calibrated 配对比较；跨图可比性主要看 axis-normalized error。

### 测试集按图型

| Chart type | Method | N | MAE | Mean axis-N error | Median axis-N error | ≤5% axis |
|---|---|---:|---:|---:|---:|---:|
| Bar | Raw | 18 | 0.472589 | 0.40% | 0.35% | 100.00% |
| Bar | Calibrated | 18 | 0.0753392 | 0.24% | 0.17% | 100.00% |
| Line | Raw | 4 | 7.55484 | 5.86% | 2.48% | 75.00% |
| Line | Calibrated | 4 | 7.47962 | 5.75% | 2.48% | 75.00% |

Bar 的测试集 Raw Geometry 全部落在轴跨度 5% 内；Line 仅 4 个测试 chart，且存在一个明显失败点，因此当前“可用”结论主要适用于简单 Bar，Line 仍需扩充数据验证。

## Geometry Error Calibrator

- 目标：`Gold - RawGeometry`（实现时先除以 axis span 以统一尺度，输出再乘回 axis span）。
- 模型：标准化特征 + Huber-ridge regression，以降低少数检测失败值对整体偏置的影响；特征只含 raw/axis/像素/Geometry 检测分数/图型结构，不含 Gold，也不使用原标签 OCR 文本、置信度或 mask 面积。
- alpha 与 Huber delta 只在训练 charts 的 5-fold grouped CV 中按 axis-normalized MAE 选择。
- 图级划分：train `65` charts，test `22` charts；交集 `0`。
- 测试集 paired chart bootstrap：raw-unit MAE 改善概率 `98.70%`，axis-normalized MAE 改善概率 `98.34%`，95% CI 如上。样本量较小时应把区间而非单一点估计作为主要不确定性提示。

## 中间结果与产物

- `results/geometry_samples.csv`：tick value/pixel_y、bar bbox 或 line point、raw value、Gold 与三种误差。
- `results/calibrated_predictions.csv`：split、predicted error、Corrected 与 Raw/Corrected 误差。
- `results/metrics_overall.csv`、`metrics_by_type.csv`、`calibration_cv.csv`、`calibration_bootstrap.csv`、`chart_split.csv`。
- `masked_images/`：每条样本的 masked chart。
- `debug_overlays/`：原图上的 tick、目标柱/点、坐标、Raw、Gold、Corrected。
- `audit/geometry_samples.jsonl`：不压平的完整中间结构；`audit/calibrator_model.json`：特征归一化与 ridge 系数。
- `logs/`：数据审计、OCR 筛选、Geometry、Calibration 与 GPU 状态日志。

## GPU 安全

初始 `nvidia-smi` 显示 A100 正被其他用户进程占用（约 90% utilization）。首个 EasyOCR `gpu=False` 进程仍短暂初始化了 414 MiB CUDA context；监控发现后只终止了本项目自己的该进程，并以 `CUDA_VISIBLE_DEVICES=""` 从 JSONL 断点恢复。复核时 `nvidia-smi` 只剩原有 PID 452389 与 453093，没有本项目进程。OCR、OpenCV、拟合与校准均在 CPU 完成，未启动 VLM/GPU compute，也没有 kill、暂停或修改任何其他用户进程。由于后续没有必需的本地 VLM 工作，本轮不创建 GPU 等待任务；若未来加入 VLM，只能使用“每 60 秒检查、连续 3 次无活跃 compute process 才启动”的独立恢复脚本。原始与整改后的快照分别保存在 `logs/gpu_initial_and_safety_check.log`、`logs/gpu_after_cuda_hidden.log`。

## 限制与下一步

1. pseudo-Gold 是原图显式标签的 OCR 双重一致读数；标签本身可能经过显示四舍五入，因此 line endpoint 与标签之间的亚像素误差并不总等于真实数据误差。
2. 数据是严格高质量子集，不代表 FinMME 所有图；当前结论仅适用于简单单轴显式标签 Bar/Line。
3. Line 样本若显著少于 Bar，应单独报告，不用 Bar 的结果替 Line 背书。
4. 若最终高质量样本少于 100，下一步先审计 FinChart-Bench/ChartBench 等公开数据是否含原始表值或稳定数据标签，再扩展数据；不要通过放宽多轴/堆叠限制凑数量。
5. Calibration 的价值只能由严格图级 test 与配对不确定性决定；本轮不据此扩展到 QA、Agent 或训练。

## Reproducibility

- Working directory: `/data/liu_jun/financial_geometry_mvp`
- Python runtime: `/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python`
- Dataset is read-only and reused in place; all新写入均位于 MVP 项目目录。
- 完整命令见 `README.md`；源码位于 `src/`。
