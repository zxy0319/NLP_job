# -*- coding: utf-8 -*-
"""坏例分析与失败归因脚本（成员 C 负责）。

读取 outputs/preds_*.jsonl，定位预测错误的样本，并按启发式规则自动归类到
典型失败类型（对应报告 §6 的错误归因），输出代表性坏例与错误方向统计，
落盘 logs/bad_cases_YYYYMMDD.log 作为工作量证据。

归类规则（优先级从高到低，每条误判样本只归入首个命中类别）：
1. negation（否定词反转）  ：文本含否定词，但模型判反 —— 情感分析最常见失败
2. long_text（长文本混淆）  ：文本过长，上下文/信息冗余导致判断失真
3. short_text（短文本信息不足）：文本过短，情感信号不充分
4. other（其他）            ：不满足上述规则的误判

否定词与长度阈值均为启发式默认值，可通过命令行参数调整；后续可据真实
坏例分布迭代规则（这也正是"发现"环节的价值）。

用法（在仓库根目录 NLP_job/ 下执行）：
    python -m src.eval.analyze_bad_cases --pred outputs/preds_baseline.jsonl
    python -m src.eval.analyze_bad_cases --pred outputs/preds_llm.jsonl --top 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .evaluate import (
    REPO_ROOT,
    STANDARD_LABELS,
    load_predictions,
    load_test_gold,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

LOG_DIR = REPO_ROOT / "logs"

# 中文否定词（单字为主，覆盖常见否定表达；"不"亦见于"不错/不过"等，属启发式噪声，可迭代）
NEGATION_WORDS = {"不", "没", "无", "非", "别", "莫", "未", "否", "勿", "毫无", "绝非", "并不"}

# 文本长度阈值（字符数），默认值；按本测试集长度分布（min=20, p5≈31, 中位=79,
# p95≈220, mean≈108）设定，可用 --short-threshold / --long-threshold 覆盖
DEFAULT_SHORT_THRESHOLD = 30
DEFAULT_LONG_THRESHOLD = 200
# 展示坏例时截断长度，避免日志过长
TRUNCATE = 60

CATEGORY_LABELS = {
    "negation": "否定词反转",
    "long_text": "长文本混淆",
    "short_text": "短文本信息不足",
    "other": "其他",
}


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"bad_cases_{datetime.now():%Y%m%d}.log"
    logger = logging.getLogger("bad_cases")
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


def _has_negation(text: str) -> bool:
    return any(w in text for w in NEGATION_WORDS)


def _categorize(text: str, short_threshold: int, long_threshold: int) -> str:
    length = len(text)
    if _has_negation(text):
        return "negation"
    if length >= long_threshold:
        return "long_text"
    if length <= short_threshold:
        return "short_text"
    return "other"


def _truncate(text: str, limit: int = TRUNCATE) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def analyze(preds: list, gold: dict, short_threshold: int, long_threshold: int, top: int) -> dict:
    """找出误判样本并按类别归类，返回统计结果。"""
    errors = []
    for p in preds:
        text = p["text"]
        if text not in gold:
            continue
        true_label = gold[text]
        pred_label = p["pred_label"]
        if pred_label == true_label:
            continue
        errors.append(
            {
                "text": text,
                "gold": true_label,
                "pred": pred_label,
                "len": len(text),
                "category": _categorize(text, short_threshold, long_threshold),
            }
        )

    by_category = defaultdict(list)
    for e in errors:
        by_category[e["category"]].append(e)

    # 错误方向统计：gold -> pred
    direction = Counter((e["gold"], e["pred"]) for e in errors)

    # 误判样本长度分布
    lengths = [e["len"] for e in errors]

    return {
        "total_preds": len([p for p in preds if p["text"] in gold]),
        "n_errors": len(errors),
        "by_category": by_category,
        "direction": direction,
        "lengths": lengths,
        "top": top,
    }


def report(logger: logging.Logger, result: dict) -> None:
    total = result["total_preds"]
    n_err = result["n_errors"]
    err_rate = n_err / total if total else 0.0

    logger.info("=" * 60)
    logger.info("坏例分析报告")
    logger.info("=" * 60)
    logger.info(f"样本总数：{total}；误判：{n_err}；错误率：{err_rate:.4f}")

    if not n_err:
        logger.info("无误判样本，跳过坏例归类")
        return

    if result["lengths"]:
        avg = sum(result["lengths"]) / len(result["lengths"])
        logger.info(f"误判样本长度：min={min(result['lengths'])} "
                    f"max={max(result['lengths'])} avg={avg:.1f}")

    logger.info("-" * 60)
    logger.info("错误方向分布（真实 -> 预测）：")
    for (g, p), c in result["direction"].most_common():
        logger.info(f"  {g} -> {p} : {c}")

    logger.info("-" * 60)
    logger.info("失败类型分布：")
    for cat, label in CATEGORY_LABELS.items():
        cnt = len(result["by_category"].get(cat, []))
        if cnt:
            logger.info(f"  {label:<8} ({cat}): {cnt}")

    for cat, label in CATEGORY_LABELS.items():
        samples = result["by_category"].get(cat, [])
        if not samples:
            continue
        logger.info("-" * 60)
        logger.info(f"【{label}】代表性坏例（前 {min(result['top'], len(samples))} 条）：")
        for e in samples[: result["top"]]:
            logger.info(f"  [{e['gold']} -> {e['pred']}] (len={e['len']}) {_truncate(e['text'])}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="坏例分析与失败归因脚本（成员 C）")
    parser.add_argument("--pred", required=True, type=str, help="预测文件路径，如 outputs/preds_baseline.jsonl")
    parser.add_argument("--short-threshold", type=int, default=DEFAULT_SHORT_THRESHOLD,
                        help=f"短文本阈值（字符数，默认 {DEFAULT_SHORT_THRESHOLD}）")
    parser.add_argument("--long-threshold", type=int, default=DEFAULT_LONG_THRESHOLD,
                        help=f"长文本阈值（字符数，默认 {DEFAULT_LONG_THRESHOLD}）")
    parser.add_argument("--top", type=int, default=5, help="每类展示的坏例条数（默认 5）")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("坏例分析脚本启动")
    logger.info(f"参数：short_threshold={args.short_threshold}，"
                f"long_threshold={args.long_threshold}，top={args.top}，"
                f"标签空间={STANDARD_LABELS}")

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在：{pred_path.as_posix()}")

    gold = load_test_gold()
    preds = load_predictions(pred_path)
    logger.info(f"读取预测文件 {pred_path.as_posix()}：{len(preds)} 条")

    result = analyze(preds, gold, args.short_threshold, args.long_threshold, args.top)
    report(logger, result)
    logger.info("坏例分析完成")


if __name__ == "__main__":
    main()
