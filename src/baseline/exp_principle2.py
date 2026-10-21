# -*- coding: utf-8 -*-
"""原理点 2 对比实验：稀疏词袋表示 vs 稠密语义向量表示（第 5 讲）。

Group 1: TF-IDF 稀疏高维特征 + LogisticRegression（复用 Baseline）
Group 2: 在训练集上就地训练 Word2Vec 词向量，对句子做平均池化得到稠密句向量，
         再接相同的 LogisticRegression 分类器。

选型说明：词向量在训练集上就地训练（gensim Word2Vec），无需下载预训练向量文件，
         普通 CPU 数秒即可完成，依赖精简、完全可复现，符合"1 分钟内一键跑完"要求。
         对比维度：特征维度、稀疏度、训练/推理耗时、Macro-F1、Accuracy。
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 保证以 python src/baseline/exp_principle2.py 方式执行时可导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jieba  # noqa: E402

from src.baseline.model import RANDOM_SEED, BaselineModel, jieba_tokenizer  # noqa: E402

jieba.setLogLevel(logging.WARNING)

TRAIN_PATH = Path("data/processed/train.jsonl")
TEST_PATH = Path("data/processed/test.jsonl")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "preds_principle2_dense.jsonl"
LOG_DIR = Path("logs")

# Word2Vec 超参数（轻量配置，CPU 秒级训练；workers=1 + seed 保证可复现）
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_EPOCHS = 10


def setup_logger() -> logging.Logger:
    """配置日志：同时输出到控制台与 logs/principle2_exp_YYYYMMDD.log。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"principle2_exp_{datetime.now():%Y%m%d}.log"
    logger = logging.getLogger("principle2")
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


def compute_metrics(labels, preds, classes):
    """计算 Macro 指标与混淆矩阵。"""
    return {
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "confusion_matrix": confusion_matrix(labels, preds, labels=classes).tolist(),
    }


def format_confusion_matrix(cm: list, classes: list) -> str:
    """混淆矩阵格式化为对齐表格（行=真实标签，列=预测标签）。"""
    header = "gold\\pred  " + "".join(f"{c:>10}" for c in classes)
    rows = [f"{c:<10}" + "".join(f"{v:>10}" for v in row) for c, row in zip(classes, cm)]
    return "\n".join([header] + rows)


def sentence_to_vec(tokens: list, w2v: Word2Vec) -> np.ndarray:
    """句向量 = 句中词向量的平均池化；无词命中时返回零向量。"""
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def run_tfidf_group(train_texts, train_labels, test_texts, test_labels, logger):
    """Group 1：TF-IDF 稀疏高维 + LogisticRegression（复用 BaselineModel）。"""
    logger.info("[Group 1] TF-IDF 稀疏高维 + LogisticRegression")
    model = BaselineModel(tokenizer="word", seed=RANDOM_SEED)
    model.train(train_texts, train_labels)
    result = model.evaluate(test_texts, test_labels)
    logger.info(
        f"  vocab_size={model.vocab_size} | train_time={model.train_seconds:.2f}s "
        f"| infer_total={model.infer_total_ms:.2f}ms | infer_avg={model.infer_avg_ms:.3f}ms"
    )
    logger.info(
        f"  precision={result['precision']:.4f} | recall={result['recall']:.4f} "
        f"| macro_f1={result['macro_f1']:.4f} | accuracy={result['accuracy']:.4f}"
    )
    logger.info(
        "  混淆矩阵 (行=gold, 列=pred):\n"
        + format_confusion_matrix(result["confusion_matrix"], result["classes"])
    )
    return {
        "representation": "TF-IDF (sparse)",
        "feature_dim": model.vocab_size,
        "train_seconds": model.train_seconds,
        "infer_avg_ms": model.infer_avg_ms,
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
    }


def run_dense_group(train_texts, train_labels, test_texts, test_labels, logger):
    """Group 2：Word2Vec 平均池化稠密句向量 + LogisticRegression。"""
    logger.info("[Group 2] Word2Vec 平均池化稠密句向量 + LogisticRegression")
    train_tokens = [jieba_tokenizer(t) for t in train_texts]
    test_tokens = [jieba_tokenizer(t) for t in test_texts]

    # 训练阶段：Word2Vec 训练 + 句向量构造 + LR 训练，统一计时
    t0 = time.perf_counter()
    w2v = Word2Vec(
        sentences=train_tokens,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=1,
        seed=RANDOM_SEED,
        epochs=W2V_EPOCHS,
    )
    X_train = np.vstack([sentence_to_vec(toks, w2v) for toks in train_tokens])
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train, train_labels)
    train_seconds = time.perf_counter() - t0
    logger.info(
        f"  Word2Vec 训练完成 | vector_size={W2V_VECTOR_SIZE} | 词表内词数={len(w2v.wv)} "
        f"| 总训练耗时(含W2V+LR)={train_seconds:.2f}s"
    )

    # 推理阶段：句向量构造 + LR 预测，统一计时
    t0 = time.perf_counter()
    X_test = np.vstack([sentence_to_vec(toks, w2v) for toks in test_tokens])
    preds = clf.predict(X_test)
    infer_total_ms = (time.perf_counter() - t0) * 1000.0
    infer_avg_ms = infer_total_ms / max(len(test_texts), 1)

    classes = clf.classes_.tolist()
    metrics = compute_metrics(test_labels, preds, classes)
    logger.info(
        f"  feature_dim={W2V_VECTOR_SIZE} | infer_total={infer_total_ms:.2f}ms "
        f"| infer_avg={infer_avg_ms:.3f}ms"
    )
    logger.info(
        f"  precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} "
        f"| macro_f1={metrics['macro_f1']:.4f} | accuracy={metrics['accuracy']:.4f}"
    )
    logger.info(
        "  混淆矩阵 (行=gold, 列=pred):\n"
        + format_confusion_matrix(metrics["confusion_matrix"], classes)
    )

    # 稠密方案预测落盘（不覆盖 preds_baseline.jsonl）
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for text, gold, pred in zip(test_texts, test_labels, preds):
            row = {
                "text": text,
                "gold_label": gold,
                "pred_label": pred,
                "latency_ms": round(infer_avg_ms, 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"  稠密方案预测已保存至 {OUTPUT_PATH.as_posix()} ({len(test_texts)} 行)")

    return {
        "representation": "Word2Vec avg (dense)",
        "feature_dim": W2V_VECTOR_SIZE,
        "train_seconds": train_seconds,
        "infer_avg_ms": infer_avg_ms,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
    }


def format_comparison_table(results: list) -> str:
    """原理点 2 对比小表格。"""
    header = (
        f"{'Representation':<22}{'FeatureDim':>11}{'TrainTime(s)':>13}"
        f"{'InferAvg(ms)':>13}{'Macro-F1':>10}{'Accuracy':>10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        lines.append(
            f"{r['representation']:<22}{r['feature_dim']:>11}{r['train_seconds']:>13.2f}"
            f"{r['infer_avg_ms']:>13.3f}{r['macro_f1']:>10.4f}{r['accuracy']:>10.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main():
    logger = setup_logger()
    logger.info("===== 原理点 2：词向量与句向量表示能力差异对比实验 =====")
    logger.info(f"Random seed: {RANDOM_SEED}")
    # jieba 词典预热，避免首次切词耗时计入训练时间，保证两组对比公平
    jieba.initialize()

    train_texts, train_labels = load_jsonl(TRAIN_PATH)
    test_texts, test_labels = load_jsonl(TEST_PATH)
    logger.info(f"训练集: {len(train_texts)} 条 | 测试集: {len(test_texts)} 条 (只读，严禁篡改)")

    r1 = run_tfidf_group(train_texts, train_labels, test_texts, test_labels, logger)
    r2 = run_dense_group(train_texts, train_labels, test_texts, test_labels, logger)

    table = format_comparison_table([r1, r2])
    logger.info("对比结果（稀疏词袋 vs 稠密语义向量）:\n" + table)
    logger.info("原理点 2 实验完成。")


if __name__ == "__main__":
    main()
