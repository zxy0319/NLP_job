# -*- coding: utf-8 -*-
"""坏例分析与失败归因脚本（成员 C 负责）。

读取 outputs/preds_*.jsonl，定位预测错误的样本，按启发式规则自动归类到
典型失败类型（对应报告 §6 的错误归因，以及任务书「≥3 类失败样例逐一归因」
的硬性要求），输出：

1. 控制台 + logs/bad_cases_YYYYMMDD.log 摘要（含代表性坏例原文与归因说明）；
2. 结构化文件 outputs/bad_cases_{model}.json —— 每条坏例含 text / gold_label /
   pred_label / category / reason，供报告 §6 直接引用。

归类规则（优先级从高到低，每条误判样本只归入首个命中类别）：
1. contrast（转折反转）   ：含转折/让步连词（但是/却/然而/虽然/不过…），
                            前后情感反转，模型未能捕捉转折语义
2. sarcasm（疑似讽刺/反语） ：含反语标记（居然/竟然/号称/所谓/呵呵…），
                            模型按字面理解
3. negation（否定词）      ：含否定词（不/没/无/非/别/莫/未/否…）
4. long_text（长文本混淆） ：文本过长，上下文/信息冗余导致判断失真
5. short_text（短文本不足） ：文本过短，情感信号不充分
6. other（其他）           ：无明显特征，疑似边界模糊或标注歧义

用法（在仓库根目录 NLP_job/ 下执行）：
    python -m src.eval.analyze_bad_cases --pred outputs/preds_baseline.jsonl
    python -m src.eval.analyze_bad_cases --pred outputs/preds_llm.jsonl --model llm --top 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 相对本文件解析仓库根目录，并支持直接 `python src/eval/analyze_bad_cases.py` 执行（无需 -m）
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluate import (
    OUTPUTS_DIR,
    STANDARD_LABELS,
    load_predictions,
    load_test_gold,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

LOG_DIR = REPO_ROOT / "logs"

# 转折/让步连词：前后情感反转的典型信号（如「虽然便宜但是很差」）
# 注意：单字「可」过于噪声（会误匹配「可能/可以」），故仅保留「可是」
TURN_WORDS = [
    "但是", "但", "却", "然而", "不过", "只是", "可是", "偏偏",
    "谁知", "结果", "反而", "虽然", "尽管", "固然", "即使", "即便", "哪怕",
]

# 反语/讽刺弱标记：口吻特殊、疑似按字面反着说（弱启发式，最终以人工复核为准）
SARCASM_WORDS = [
    "呵呵", "哈哈", "居然", "竟然", "号称", "所谓", "敢情", "不愧是",
    "多谢", "佩服", "好一个", "还真是", "真行", "真有", "神了",
]

# 中文否定词（单字为主；「不」亦见于「不错/不过」等，属启发式噪声，可迭代）
NEGATION_WORDS = ["不", "没", "无", "非", "别", "莫", "未", "否", "勿", "毫无", "绝不", "并不"]

# 文本长度阈值（字符数）；按本测试集长度分布（min=20, p5≈31, 中位=79, p95≈220）
DEFAULT_SHORT_THRESHOLD = 30
DEFAULT_LONG_THRESHOLD = 200
# 展示坏例时截断长度
TRUNCATE = 60

CATEGORY_LABELS = {
    "contrast": "转折反转",
    "sarcasm": "疑似讽刺/反语",
    "negation": "否定词误判",
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


def _match(word_list: list, text: str):
    """返回 text 中命中的第一个词，未命中返回 None。"""
    for w in word_list:
        if w in text:
            return w
    return None


def _categorize(text: str, short_threshold: int, long_threshold: int) -> tuple:
    """返回 (category, matched_word)，按优先级归类。"""
    length = len(text)

    turn = _match(TURN_WORDS, text)
    if turn:
        return "contrast", turn

    sarc = _match(SARCASM_WORDS, text)
    if sarc:
        return "sarcasm", sarc

    neg = _match(NEGATION_WORDS, text)
    if neg:
        return "negation", neg

    if length >= long_threshold:
        return "long_text", str(length)
    if length <= short_threshold:
        return "short_text", str(length)
    return "other", ""


def _make_reason(category: str, matched: str, length: int) -> str:
    """根据类别生成错误机理归因说明。"""
    if category == "contrast":
        return f"含转折连词「{matched}」，前后情感反转，模型未能捕捉转折语义"
    if category == "sarcasm":
        return f"含反语标记「{matched}」，疑似讽刺/口吻特殊，模型按字面理解"
    if category == "negation":
        return f"含否定词「{matched}」，模型未能正确处理否定范围"
    if category == "long_text":
        return f"文本过长（{length}字），上下文信息冗余导致混淆"
    if category == "short_text":
        return f"文本过短（{length}字），情感信号不足"
    return "无明显特征，疑似边界模糊或标注歧义"


def _truncate(text: str, limit: int = TRUNCATE) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _infer_model(pred_path: Path) -> str:
    """从预测文件名推断模型名：preds_baseline -> baseline，preds_llm -> llm。"""
    name = pred_path.stem.lower()
    if "baseline" in name:
        return "baseline"
    if "llm" in name:
        return "llm"
    return name.replace("preds_", "") or name


def analyze(preds: list, gold: dict, short_threshold: int, long_threshold: int) -> dict:
    """找出误判样本并归类，返回含明细的统计结果。"""
    errors = []
    for p in preds:
        text = p["text"]
        if text not in gold:
            continue
        true_label = gold[text]
        pred_label = p["pred_label"]
        if pred_label == true_label:
            continue
        category, matched = _categorize(text, short_threshold, long_threshold)
        errors.append(
            {
                "text": text,
                "gold_label": true_label,
                "pred_label": pred_label,
                "len": len(text),
                "category": category,
                "matched": matched,
                "reason": _make_reason(category, matched, len(text)),
            }
        )

    by_category = defaultdict(list)
    for e in errors:
        by_category[e["category"]].append(e)

    direction = Counter((e["gold_label"], e["pred_label"]) for e in errors)

    return {
        "total_preds": len([p for p in preds if p["text"] in gold]),
        "n_errors": len(errors),
        "by_category": by_category,
        "direction": direction,
        "errors": errors,
    }


def report(logger: logging.Logger, result: dict, top: int) -> None:
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
        logger.info(f"【{label}】代表性坏例（前 {min(top, len(samples))} 条）：")
        for e in samples[:top]:
            logger.info(f"  [{e['gold_label']} -> {e['pred_label']}] (len={e['len']}) {_truncate(e['text'])}")
            logger.info(f"      归因：{e['reason']}")

    logger.info("=" * 60)


def save_json(result: dict, model: str, pred_path: Path, top: int) -> Path:
    """将坏例结果保存为结构化 JSON，供报告 §6 引用。"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"bad_cases_{model}.json"

    examples = [
        {
            "text": e["text"],
            "gold_label": e["gold_label"],
            "pred_label": e["pred_label"],
            "category": e["category"],
            "category_label": CATEGORY_LABELS[e["category"]],
            "reason": e["reason"],
        }
        for e in result["errors"]
    ]

    payload = {
        "model": model,
        "source": pred_path.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "labels": STANDARD_LABELS,
        "total_samples": result["total_preds"],
        "total_errors": result["n_errors"],
        "error_rate": round(result["n_errors"] / result["total_preds"], 4) if result["total_preds"] else 0.0,
        "error_direction": {
            f"{g}->{p}": c for (g, p), c in result["direction"].most_common()
        },
        "categories": {
            cat: {
                "label": CATEGORY_LABELS[cat],
                "count": len(result["by_category"].get(cat, [])),
            }
            for cat in CATEGORY_LABELS
        },
        "examples": examples,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="坏例分析与失败归因脚本（成员 C）")
    parser.add_argument("--pred", required=True, type=str,
                        help="预测文件路径，如 outputs/preds_baseline.jsonl")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名，用于命名输出文件（默认从文件名推断）")
    parser.add_argument("--short-threshold", type=int, default=DEFAULT_SHORT_THRESHOLD,
                        help=f"短文本阈值（默认 {DEFAULT_SHORT_THRESHOLD}）")
    parser.add_argument("--long-threshold", type=int, default=DEFAULT_LONG_THRESHOLD,
                        help=f"长文本阈值（默认 {DEFAULT_LONG_THRESHOLD}）")
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

    model = args.model or _infer_model(pred_path)
    gold = load_test_gold()
    preds = load_predictions(pred_path)
    logger.info(f"读取预测文件 {pred_path.as_posix()}：{len(preds)} 条；模型：{model}")

    result = analyze(preds, gold, args.short_threshold, args.long_threshold)
    report(logger, result, args.top)

    out_path = save_json(result, model, pred_path, args.top)
    logger.info(f"坏例结构化结果已保存：{out_path.as_posix()}（共 {result['n_errors']} 条）")
    logger.info("坏例分析完成")


if __name__ == "__main__":
    main()
