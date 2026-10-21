# -*- coding: utf-8 -*-
"""统一评测脚本（成员 C 负责）。

功能：
1. 读取 outputs/ 下各方案的预测文件（JSONL），与 data/processed/test.jsonl 的
   真实标签对齐后，计算 Precision / Recall / Macro-F1 / Accuracy 与混淆矩阵；
2. 汇总各样本 latency_ms，统计平均/最小/最大延迟与吞吐；
3. 将评测结果写入 logs/eval_YYYYMMDD.log（含时间戳、随机种子、样本数等必含字段）；
4. 支持横向对比：--compare 自动发现 outputs/preds_*.jsonl 并生成 Markdown 对比表；
5. 支持 --selfcheck：用真实测试集标签构造「上界(oracle)」与「多数类基线」，
   验证评测流水线正确性（oracle 应得满分，多数类基线约为类别占比）；
6. 支持 --plot：为 --pred 的单个方案绘制混淆矩阵热力图（PNG，存到 outputs/）。

用法（在仓库根目录 NLP_job/ 下执行）：
    python -m src.eval.evaluate --pred outputs/preds_baseline.jsonl
    python -m src.eval.evaluate --pred outputs/preds_baseline.jsonl --plot
    python -m src.eval.evaluate --compare
    python -m src.eval.evaluate --selfcheck
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

from .metrics import compute_metrics
from .visualize import plot_confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_SEED = 42
STANDARD_LABELS = ["positive", "negative"]

# 相对本文件解析仓库根目录，避免依赖调用时的 cwd
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = REPO_ROOT / "data" / "processed" / "test.jsonl"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOG_DIR = REPO_ROOT / "logs"


def setup_logger(name: str = "eval") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"eval_{datetime.now():%Y%m%d}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Log file: {log_file.as_posix()}")
    return logger


def load_test_gold() -> dict:
    """读取冻结测试集，返回 {text: label}（真实标签为权威来源）。"""
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"测试集不存在：{TEST_FILE.as_posix()}")
    gold = {}
    with TEST_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold[item["text"]] = item["label"]
    return gold


def load_predictions(path: Path) -> list:
    """读取预测文件并校验字段，返回 [{text, gold_label, pred_label, latency_ms}]。"""
    preds = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.as_posix()} line {line_no}: invalid JSON ({e})")
            for field in ("text", "gold_label", "pred_label"):
                if field not in item:
                    raise ValueError(f"{path.as_posix()} line {line_no}: missing '{field}'")
            preds.append(
                {
                    "text": item["text"],
                    "gold_label": item["gold_label"],
                    "pred_label": item["pred_label"],
                    "latency_ms": item.get("latency_ms"),
                }
            )
    return preds


def evaluate_predictions(preds: list, gold: dict, logger: logging.Logger, name: str) -> dict:
    """对齐预测与真实标签，计算指标并返回结果字典。"""
    y_true, y_pred, latency = [], [], []
    gold_mismatch = 0
    missing_in_test = 0
    invalid_pred = 0

    for p in preds:
        text = p["text"]
        if text not in gold:
            missing_in_test += 1
            continue
        y_true.append(gold[text])  # 以冻结测试集标签为准
        pred_label = p["pred_label"]
        y_pred.append(pred_label)
        if p["gold_label"] != gold[text]:
            gold_mismatch += 1
        if pred_label not in STANDARD_LABELS:
            invalid_pred += 1
        if isinstance(p["latency_ms"], (int, float)):
            latency.append(float(p["latency_ms"]))

    if gold_mismatch:
        logger.warning(
            f"[{name}] {gold_mismatch} 条样本的预测文件 gold_label 与冻结测试集不一致，"
            f"已以 test.jsonl 标签为准"
        )
    if missing_in_test:
        logger.warning(f"[{name}] {missing_in_test} 条预测文本不在测试集中，已跳过")
    if invalid_pred:
        logger.warning(f"[{name}] {invalid_pred} 条预测标签不在 {STANDARD_LABELS}，已按非法处理")

    metrics = compute_metrics(y_true, y_pred, STANDARD_LABELS)

    latency_stats = None
    if latency:
        latency_stats = {
            "mean_ms": statistics.fmean(latency),
            "min_ms": min(latency),
            "max_ms": max(latency),
            "count": len(latency),
        }

    return {
        "name": name,
        "metrics": metrics,
        "latency_stats": latency_stats,
        "gold_mismatch": gold_mismatch,
        "missing_in_test": missing_in_test,
        "invalid_pred": invalid_pred,
    }


def format_report(result: dict) -> str:
    m = result["metrics"]
    lines = [f"===== 评测报告：{result['name']} ====="]
    lines.append(f"样本数：{m['n']}")
    lines.append(f"Accuracy     : {m['accuracy']:.4f}")
    lines.append(f"Macro-Precision: {m['macro_precision']:.4f}")
    lines.append(f"Macro-Recall   : {m['macro_recall']:.4f}")
    lines.append(f"Macro-F1       : {m['macro_f1']:.4f}")
    lines.append("-" * 40)
    lines.append(f"{'label':<10}{'P':>8}{'R':>8}{'F1':>8}{'support':>10}")
    for label, r in m["per_class"].items():
        lines.append(
            f"{label:<10}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}{r['support']:>10}"
        )
    lines.append("-" * 40)
    lines.append("混淆矩阵（行=真实，列=预测）：")
    header = " " * 10 + "".join(f"{c:>12}" for c in m["labels"])
    lines.append(header)
    for t in m["labels"]:
        row = f"{t:<10}" + "".join(f"{m['confusion_matrix'][t].get(p, 0):>12}" for p in m["labels"])
        lines.append(row)
    if result["latency_stats"]:
        ls = result["latency_stats"]
        lines.append("-" * 40)
        lines.append(
            f"延迟统计：mean={ls['mean_ms']:.2f}ms  min={ls['min_ms']:.2f}ms  "
            f"max={ls['max_ms']:.2f}ms  (n={ls['count']})"
        )
    return "\n".join(lines)


def build_comparison_table(results: list) -> str:
    """生成多方案横向对比 Markdown 表。"""
    if not results:
        return "（无可用结果）"
    header = "| 方案 | Accuracy | Macro-P | Macro-R | Macro-F1 | 平均延迟(ms) | 样本数 |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for r in results:
        m = r["metrics"]
        lat = f"{r['latency_stats']['mean_ms']:.2f}" if r["latency_stats"] else "-"
        rows.append(
            f"| {r['name']} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} "
            f"| {m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {lat} | {m['n']} |"
        )
    return "\n".join(rows)


def run_selfcheck(logger: logging.Logger) -> list:
    """用真实测试集标签验证评测流水线：oracle 上界 + 多数类基线。"""
    gold = load_test_gold()
    logger.info(f"SelfCheck: 读取测试集 {len(gold)} 条真实标签")
    logger.info(f"SelfCheck: 随机种子 seed={RANDOM_SEED}（本次不涉及随机抽样）")

    # 1) oracle 上界：预测恒等于真实标签，应得到满分
    oracle_preds = [
        {"text": t, "gold_label": lb, "pred_label": lb, "latency_ms": 0.0}
        for t, lb in gold.items()
    ]
    # 2) 多数类基线：全部预测为 positive（测试集中占比最高类）
    majority_preds = [
        {"text": t, "gold_label": lb, "pred_label": "positive", "latency_ms": 0.0}
        for t, lb in gold.items()
    ]

    results = [
        evaluate_predictions(oracle_preds, gold, logger, "selfcheck_oracle(上界)"),
        evaluate_predictions(majority_preds, gold, logger, "selfcheck_majority(多数类基线)"),
    ]

    for r in results:
        logger.info("\n" + format_report(r))

    logger.info("\n" + build_comparison_table(results))
    return results


def run_single(path_str: str, logger: logging.Logger, plot: bool = False) -> list:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在：{path.as_posix()}")
    gold = load_test_gold()
    preds = load_predictions(path)
    logger.info(f"读取预测文件 {path.as_posix()}：{len(preds)} 条")
    result = evaluate_predictions(preds, gold, logger, path.name)
    logger.info("\n" + format_report(result))
    if plot:
        stem = path.stem  # 如 preds_baseline
        fig_path = OUTPUTS_DIR / f"confusion_matrix_{stem}.png"
        try:
            plot_confusion_matrix(
                result["metrics"]["confusion_matrix"],
                result["metrics"]["labels"],
                fig_path,
                title=f"Confusion Matrix - {stem}",
            )
            logger.info(f"混淆矩阵图已保存：{fig_path.as_posix()}")
        except ImportError as e:
            logger.warning(f"未安装 matplotlib，跳过出图：{e}")
    return [result]


def run_compare(logger: logging.Logger) -> list:
    if not OUTPUTS_DIR.exists():
        logger.warning(f"outputs/ 目录不存在：{OUTPUTS_DIR.as_posix()}")
        return []
    pred_files = sorted(OUTPUTS_DIR.glob("preds_*.jsonl"))
    if not pred_files:
        logger.warning("outputs/ 下未发现任何 preds_*.jsonl，暂无方案可对比")
        return []
    gold = load_test_gold()
    results = []
    for pf in pred_files:
        preds = load_predictions(pf)
        logger.info(f"读取 {pf.as_posix()}：{len(preds)} 条")
        results.append(evaluate_predictions(preds, gold, logger, pf.name))
    logger.info("\n" + build_comparison_table(results))
    return results


def main():
    parser = argparse.ArgumentParser(description="统一评测脚本（成员 C）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred", "--pred_path", dest="pred", type=str,
                       help="评测单个预测文件，如 outputs/preds_baseline.jsonl")
    group.add_argument("--compare", action="store_true", help="自动对比 outputs/ 下所有 preds_*.jsonl")
    group.add_argument("--selfcheck", action="store_true", help="用真实标签验证评测流水线")
    parser.add_argument("--plot", action="store_true", help="为 --pred 生成混淆矩阵 PNG 图（存到 outputs/）")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("统一评测脚本启动")
    logger.info(f"随机种子 seed={RANDOM_SEED}；标签空间={STANDARD_LABELS}")

    if args.selfcheck:
        run_selfcheck(logger)
    elif args.compare:
        run_compare(logger)
    else:
        run_single(args.pred, logger, plot=args.plot)

    logger.info("评测完成")


if __name__ == "__main__":
    main()
