# -*- coding: utf-8 -*-
"""非大模型基线（Non-LLM Baseline）运行入口脚本。

流程一：默认 jieba 词级别分词训练 TF-IDF + LogisticRegression，
        在 test.jsonl 上批量推理，指标写入日志，
        预测逐行保存到 outputs/preds_baseline.jsonl；
流程二（课程原理点 1）：对比 char-level（分字）与 word-level（jieba 分词）
        的词表大小、训练耗时、测试集 Macro-F1 与 Accuracy，
        输出对比小表格，为报告第 4/5 节提供直接证据。

注意：本脚本只读取 test.jsonl 用于评估，严禁改动或泄漏（不写入、不打乱）。
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 保证以 python src/baseline/run_baseline.py 方式执行时可导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jieba  # noqa: E402

from src.baseline.model import RANDOM_SEED, BaselineModel  # noqa: E402

TRAIN_PATH = Path("data/processed/train.jsonl")
TEST_PATH = Path("data/processed/test.jsonl")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "preds_baseline.jsonl"
LOG_DIR = Path("logs")
DEFAULT_TOKENIZER = "word"


def setup_logger() -> logging.Logger:
    """配置日志：同时输出到控制台与 logs/baseline_run_YYYYMMDD.log。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"baseline_run_{datetime.now():%Y%m%d}.log"
    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_jsonl(path: Path):
    """读取统一格式 jsonl，返回 (texts, labels)；非法行抛出明确异常。"""
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.as_posix()} 第 {line_no} 行非法 JSON: {e}")
            if "text" not in item or "label" not in item:
                raise ValueError(f"{path.as_posix()} 第 {line_no} 行缺少 text/label 字段")
            texts.append(str(item["text"]))
            labels.append(str(item["label"]))
    return texts, labels


def format_confusion_matrix(cm: list, classes: list) -> str:
    """混淆矩阵格式化为对齐表格（行=真实标签，列=预测标签）。"""
    header = "gold\\pred  " + "".join(f"{c:>10}" for c in classes)
    rows = [f"{c:<10}" + "".join(f"{v:>10}" for v in row) for c, row in zip(classes, cm)]
    return "\n".join([header] + rows)


def run_pipeline(logger: logging.Logger, tokenizer: str, save_predictions: bool = False) -> dict:
    """训练 + 评估一条完整流水线，返回供对比表格使用的结果摘要。"""
    logger.info(f"Initialized baseline with random seed: {RANDOM_SEED}")
    train_texts, train_labels = load_jsonl(TRAIN_PATH)
    logger.info(f"Loaded train data: {len(train_texts)} samples from {TRAIN_PATH.as_posix()}")

    model = BaselineModel(tokenizer=tokenizer, seed=RANDOM_SEED)
    logger.info(f"Training model with tokenizer='{tokenizer}' ...")
    model.train(train_texts, train_labels)
    logger.info(
        f"Training done | tokenizer={tokenizer} | vocab_size={model.vocab_size} "
        f"| train_time={model.train_seconds:.2f}s"
    )

    # 只读 test.jsonl 用于评估，严禁改动或混入训练
    test_texts, test_labels = load_jsonl(TEST_PATH)
    logger.info(f"Loaded test data: {len(test_texts)} samples from {TEST_PATH.as_posix()} (read-only)")

    result = model.evaluate(test_texts, test_labels)
    logger.info(
        f"Test inference | total_latency={model.infer_total_ms:.2f} ms "
        f"| avg_per_sample={model.infer_avg_ms:.3f} ms"
    )
    logger.info(
        f"Metrics | precision(macro)={result['precision']:.4f} | recall(macro)={result['recall']:.4f} "
        f"| macro_f1={result['macro_f1']:.4f} | accuracy={result['accuracy']:.4f}"
    )
    logger.info(
        "Confusion matrix (rows=gold, cols=pred):\n"
        + format_confusion_matrix(result["confusion_matrix"], result["classes"])
    )

    if save_predictions:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            for text, gold, pred in zip(test_texts, test_labels, result["preds"]):
                row = {
                    "text": text,
                    "gold_label": gold,
                    "pred_label": pred,
                    "latency_ms": round(model.infer_avg_ms, 3),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Predictions saved to {OUTPUT_PATH.as_posix()} ({len(test_texts)} rows)")

    return {
        "tokenizer": tokenizer,
        "vocab_size": model.vocab_size,
        "train_seconds": model.train_seconds,
        "infer_total_ms": model.infer_total_ms,
        "infer_avg_ms": model.infer_avg_ms,
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
    }


def format_comparison_table(results: list) -> str:
    """原理点 1 对比小表格（char vs word）。"""
    header = (
        f"{'Tokenizer':<12}{'VocabSize':>10}{'TrainTime(s)':>13}"
        f"{'InferAvg(ms)':>13}{'Macro-F1':>10}{'Accuracy':>10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        lines.append(
            f"{r['tokenizer']:<12}{r['vocab_size']:>10}{r['train_seconds']:>13.2f}"
            f"{r['infer_avg_ms']:>13.3f}{r['macro_f1']:>10.4f}{r['accuracy']:>10.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main():
    logger = setup_logger()
    logger.info("===== Day 2: Non-LLM Baseline (TF-IDF + LogisticRegression) =====")
    # jieba 词典预热，避免首次切词的词典加载耗时计入训练时间，保证对比公平
    jieba.initialize()

    # 流程一：默认分词配置（jieba word-level）训练 + 测试集推理 + 预测落盘
    word_result = run_pipeline(logger, DEFAULT_TOKENIZER, save_predictions=True)

    # 流程二：课程原理点 1 对比实验（char-level 分字 vs word-level 分词）
    logger.info("===== 原理点 1 对比实验：分字(char) vs jieba分词(word) =====")
    char_result = run_pipeline(logger, "char")
    table = format_comparison_table([char_result, word_result])
    logger.info("对比结果（分词粒度对特征维度与分类效果的影响）:\n" + table)

    logger.info("Baseline pipeline finished successfully.")


if __name__ == "__main__":
    main()
