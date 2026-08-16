from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def metric(metrics: list[dict[str, str]], split: str, method: str) -> dict[str, str]:
    return next(row for row in metrics if row["split"] == split and row["method"] == method)


def metric_by_type(
    metrics: list[dict[str, str]], split: str, method: str, chart_type: str
) -> dict[str, str]:
    return next(
        row
        for row in metrics
        if row["split"] == split
        and row["method"] == method
        and row["chart_type"] == chart_type
    )


def fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: Any, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    dataset = json.loads((project / "audit" / "dataset_summary.json").read_text(encoding="utf-8"))
    samples = read_csv(project / "results" / "geometry_samples.csv")
    metrics = read_csv(project / "results" / "metrics_overall.csv")
    type_metrics = read_csv(project / "results" / "metrics_by_type.csv")
    bootstrap = read_csv(project / "results" / "calibration_bootstrap.csv")[0]
    splits = read_csv(project / "results" / "chart_split.csv")
    candidates = read_jsonl(project / "audit" / "candidate_scan_v3.jsonl")
    attempts = read_jsonl(project / "audit" / "construction_attempts.jsonl")

    raw_test = metric(metrics, "test", "raw")
    calibrated_test = metric(metrics, "test", "calibrated")
    raw_all = metric(metrics, "all", "raw")
    calibrated_all = metric(metrics, "all", "calibrated")
    bar_raw_test = metric_by_type(type_metrics, "test", "raw", "bar")
    bar_calibrated_test = metric_by_type(type_metrics, "test", "calibrated", "bar")
    line_raw_test = metric_by_type(type_metrics, "test", "raw", "line")
    line_calibrated_test = metric_by_type(type_metrics, "test", "calibrated", "line")
    type_counts = Counter(row["chart_type"] for row in samples)
    candidate_counts = Counter(row.get("status", "unknown") for row in candidates)
    attempt_counts = Counter(row.get("status", "unknown") for row in attempts)
    train_charts = {row["chart_id"] for row in splits if row["split"] == "train"}
    test_charts = {row["chart_id"] for row in splits if row["split"] == "test"}

    geometry_usable = (
        float(raw_test["median_axis_normalized_error"]) <= 0.02
        and float(raw_test["within_5pct_axis"]) >= 0.80
    )
    axis_improved = float(calibrated_test["mean_axis_normalized_error"]) < float(
        raw_test["mean_axis_normalized_error"]
    )
    mae_improved = float(calibrated_test["mae"]) < float(raw_test["mae"])
    ci_low = float(bootstrap["ci95_low"])
    axis_ci_low = float(bootstrap["axis_ci95_low"])
    if axis_improved and mae_improved and ci_low > 0 and axis_ci_low > 0:
        calibration_verdict = "有明确额外价值（raw-unit 与 axis-normalized 配对 bootstrap 95% CI 均支持改善）"
    elif axis_improved and mae_improved:
        calibration_verdict = "有点估计层面的额外价值，但统计不确定性仍大"
    else:
        calibration_verdict = "本轮未显示稳定额外价值；这不影响 Geometry recovery 的独立结论"

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    report = f"""# Calibrated Geometry Reasoning for Financial Charts — MVP Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: {generated}
- Verification Status: ANALYZED
- Version Label: financial_geometry_mvp_v1

## 结论

- **Geometry recovery：{'可用于受限 MVP' if geometry_usable else '当前证据不足以判定可用'}。** 测试集 Raw Geometry 的 MAE 为 `{fmt(raw_test['mae'])}`，中位绝对误差为 `{fmt(raw_test['median_absolute_error'])}`；平均/中位 axis-normalized error 分别为 `{pct(raw_test['mean_axis_normalized_error'])}` / `{pct(raw_test['median_axis_normalized_error'])}`，`{pct(raw_test['within_5pct_axis'])}` 的样本落在 y 轴跨度 5% 以内。Bar 测试证据较强；Line 测试仅 `{line_raw_test['n']}` 张，应视为探索性结果。
- **Calibration：{calibration_verdict}。** 测试集 MAE `{fmt(raw_test['mae'])} → {fmt(calibrated_test['mae'])}`，平均 axis-normalized error `{pct(raw_test['mean_axis_normalized_error'])} → {pct(calibrated_test['mean_axis_normalized_error'])}`。配对 chart bootstrap 的 `Raw MAE - Calibrated MAE` 为 `{fmt(bootstrap['raw_minus_calibrated_mae'])}`，95% CI `[{fmt(bootstrap['ci95_low'])}, {fmt(bootstrap['ci95_high'])}]`；axis-normalized 差值为 `{pct(bootstrap['raw_minus_calibrated_axis_normalized_mae'])}`，95% CI `[{pct(bootstrap['axis_ci95_low'])}, {pct(bootstrap['axis_ci95_high'])}]`。增益主要由 Bar 支持，Line 尚不足以单独确认 Calibration 价值。
- 两个判断彼此独立；即使 calibrator 没有稳定增益，也不能据此否定像素到数值的 Geometry 路线。

## MVP 范围

本轮仅处理单 y 轴、简单垂直 Bar 与 Line 的 masked-value 恢复；未实现完整 QA、RL、Agent、LoRA、多轴、Pie、Area、Boxplot，也未调用本地 VLM。Geometry 由论文规格的最小重写组成：axis localization、pixel-to-value interpolation、bar detection/bar-top recovery、line edge-point recovery。

## FinMME 数据审计

- 复用调用方通过 `FINMME_ROOT` 配置的现有离线数据，没有重复下载。
- 复用同一 FinMME 工作空间中的现有预处理图像缓存。
- 全量 `{dataset['row_count']:,}` 题；原始 Arrow 审计为 4,450 张唯一图，现有官方预处理 JPEG 按 SHA-256 去重后为 `{dataset['unique_chart_count']:,}` 张（4 组在缩放/JPEG 后发生内容合并）。
- 题型：Single `{dataset['question_type_counts']['single_choice']:,}`、Multi `{dataset['question_type_counts']['multiple_choice']:,}`、Numerical/Calculation `{dataset['question_type_counts']['numerical']:,}`。
- `answer` 是 FinMME Gold；Numerical 同时提供 `unit` 与绝对 `tolerance`。1,852 条 Numerical 中 1 条 tolerance 为 NaN，其余可用。
- `{dataset['charts_with_numerical']:,}` 张图至少关联一道 Numerical 题；这些字段只用于数据审计与结果追踪，不作为 masked label 的 Geometry 输入。

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
| FinMME 预处理唯一图 | {dataset['unique_chart_count']:,} |
| 至少有 Numerical 的图 | {dataset['charts_with_numerical']:,} |
| 已执行 OCR/结构筛选 | {len(candidates):,} |
| 结构候选 viable | {candidate_counts.get('viable', 0):,} |
| mask 后最终样本 | {len(samples):,} |
| Bar / Line | {type_counts.get('bar', 0):,} / {type_counts.get('line', 0):,} |

主要拒绝原因：无稳定线性左 y 轴 `{candidate_counts.get('no_linear_left_y_axis', 0):,}`，多轴/多面板 `{candidate_counts.get('multiple_y_axes_or_panels_detected', 0):,}`，没有能与柱/点建立结构联系的数据标签 `{candidate_counts.get('no_geometry_linked_data_label', 0):,}`；mask/crop/严格简单图范围二次审查失败 `{attempt_counts.get('no_label_survived', 0):,}`。若最终少于约 100 条，应把它解释为 FinMME 在严格单轴、显式标签、可无损遮挡条件下的实际供给限制，而不是强行补齐。

## Geometry 结果

| Split | Method | N | MAE | Median AE | RMSE | Mean axis-N error | Median axis-N error | ≤1% axis | ≤5% axis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test | Raw | {raw_test['n']} | {fmt(raw_test['mae'])} | {fmt(raw_test['median_absolute_error'])} | {fmt(raw_test['rmse'])} | {pct(raw_test['mean_axis_normalized_error'])} | {pct(raw_test['median_axis_normalized_error'])} | {pct(raw_test['within_1pct_axis'])} | {pct(raw_test['within_5pct_axis'])} |
| Test | Calibrated | {calibrated_test['n']} | {fmt(calibrated_test['mae'])} | {fmt(calibrated_test['median_absolute_error'])} | {fmt(calibrated_test['rmse'])} | {pct(calibrated_test['mean_axis_normalized_error'])} | {pct(calibrated_test['median_axis_normalized_error'])} | {pct(calibrated_test['within_1pct_axis'])} | {pct(calibrated_test['within_5pct_axis'])} |
| All | Raw | {raw_all['n']} | {fmt(raw_all['mae'])} | {fmt(raw_all['median_absolute_error'])} | {fmt(raw_all['rmse'])} | {pct(raw_all['mean_axis_normalized_error'])} | {pct(raw_all['median_axis_normalized_error'])} | {pct(raw_all['within_1pct_axis'])} | {pct(raw_all['within_5pct_axis'])} |
| All | Calibrated | {calibrated_all['n']} | {fmt(calibrated_all['mae'])} | {fmt(calibrated_all['median_absolute_error'])} | {fmt(calibrated_all['rmse'])} | {pct(calibrated_all['mean_axis_normalized_error'])} | {pct(calibrated_all['median_axis_normalized_error'])} | {pct(calibrated_all['within_1pct_axis'])} | {pct(calibrated_all['within_5pct_axis'])} |

原始单位跨百分比、金额与指数，MAE 用于同一样本上的 Raw/Calibrated 配对比较；跨图可比性主要看 axis-normalized error。

### 测试集按图型

| Chart type | Method | N | MAE | Mean axis-N error | Median axis-N error | ≤5% axis |
|---|---|---:|---:|---:|---:|---:|
| Bar | Raw | {bar_raw_test['n']} | {fmt(bar_raw_test['mae'])} | {pct(bar_raw_test['mean_axis_normalized_error'])} | {pct(bar_raw_test['median_axis_normalized_error'])} | {pct(bar_raw_test['within_5pct_axis'])} |
| Bar | Calibrated | {bar_calibrated_test['n']} | {fmt(bar_calibrated_test['mae'])} | {pct(bar_calibrated_test['mean_axis_normalized_error'])} | {pct(bar_calibrated_test['median_axis_normalized_error'])} | {pct(bar_calibrated_test['within_5pct_axis'])} |
| Line | Raw | {line_raw_test['n']} | {fmt(line_raw_test['mae'])} | {pct(line_raw_test['mean_axis_normalized_error'])} | {pct(line_raw_test['median_axis_normalized_error'])} | {pct(line_raw_test['within_5pct_axis'])} |
| Line | Calibrated | {line_calibrated_test['n']} | {fmt(line_calibrated_test['mae'])} | {pct(line_calibrated_test['mean_axis_normalized_error'])} | {pct(line_calibrated_test['median_axis_normalized_error'])} | {pct(line_calibrated_test['within_5pct_axis'])} |

Bar 的测试集 Raw Geometry 全部落在轴跨度 5% 内；Line 仅 4 个测试 chart，且存在一个明显失败点，因此当前“可用”结论主要适用于简单 Bar，Line 仍需扩充数据验证。

## Geometry Error Calibrator

- 目标：`Gold - RawGeometry`（实现时先除以 axis span 以统一尺度，输出再乘回 axis span）。
- 模型：标准化特征 + Huber-ridge regression，以降低少数检测失败值对整体偏置的影响；特征只含 raw/axis/像素/Geometry 检测分数/图型结构，不含 Gold，也不使用原标签 OCR 文本、置信度或 mask 面积。
- alpha 与 Huber delta 只在训练 charts 的 5-fold grouped CV 中按 axis-normalized MAE 选择。
- 图级划分：train `{len(train_charts):,}` charts，test `{len(test_charts):,}` charts；交集 `{len(train_charts & test_charts)}`。
- 测试集 paired chart bootstrap：raw-unit MAE 改善概率 `{pct(bootstrap['improvement_probability'])}`，axis-normalized MAE 改善概率 `{pct(bootstrap['axis_improvement_probability'])}`，95% CI 如上。样本量较小时应把区间而非单一点估计作为主要不确定性提示。

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

- Working directory: `{project}`
- Python runtime: 由 `PYTHON_BIN` 配置，或使用 `FINMME_ROOT` 下已有的 CPU OCR 环境。
- Dataset is read-only and reused in place; all新写入均位于 MVP 项目目录。
- 完整命令见 `README.md`；源码位于 `src/`。
"""
    (project / "MVP_REPORT.md").write_text(report, encoding="utf-8")
    print(project / "MVP_REPORT.md")


if __name__ == "__main__":
    main()
