# -*- coding: utf-8 -*-
"""非大模型基线（Non-LLM Baseline）分类模型封装。

采用 TF-IDF 特征提取 + LogisticRegression 分类器，
支持两种中文分词粒度切换（对应课程原理点 1）：
- word：jieba 分词（词级别）
- char ：逐字切分（字级别）

分词粒度与词表选择直接影响特征维度（稀疏度）与最终分类效果，
对比实验见 run_baseline.py 的 run_comparison 流程。
"""

import logging
import time

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 抑制 jieba 构建词典时的 stderr 输出，保持日志整洁
jieba.setLogLevel(logging.WARNING)

RANDOM_SEED = 42


def char_tokenizer(text: str) -> list:
    """字级别分词：逐字切分，过滤空白字符。"""
    return [ch for ch in text if not ch.isspace()]


def jieba_tokenizer(text: str) -> list:
    """词级别分词：jieba 精确模式切词，过滤空白 token。"""
    return [w for w in jieba.lcut(text) if w.strip()]


TOKENIZERS = {"word": jieba_tokenizer, "char": char_tokenizer}


class BaselineModel:
    """TF-IDF + LogisticRegression 基线分类模型。

    Attributes:
        train_seconds: 最近一次训练耗时（秒），供日志记录。
        infer_total_ms: 最近一次批量推理总延时（毫秒）。
        infer_avg_ms: 最近一次推理平均单条延迟（毫秒），均摊自总延时。
    """

    def __init__(self, tokenizer: str = "word", seed: int = RANDOM_SEED):
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"未知分词器: {tokenizer}，可选: {list(TOKENIZERS)}")
        self.tokenizer_name = tokenizer
        self.tokenizer = TOKENIZERS[tokenizer]
        self.seed = seed
        self.vectorizer = None
        self.clf = None
        self.train_seconds = 0.0
        self.infer_total_ms = 0.0
        self.infer_avg_ms = 0.0

    @property
    def vocab_size(self) -> int:
        """词表大小，即 TF-IDF 特征维度。"""
        return len(self.vectorizer.vocabulary_)

    def train(self, texts: list, labels: list) -> None:
        """拟合 TF-IDF 词表并训练逻辑回归分类器。"""
        t0 = time.perf_counter()
        # 传自定义 tokenizer 时必须显式置空 token_pattern，避免被覆盖产生警告
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=None)
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        self.clf.fit(X, labels)
        self.train_seconds = time.perf_counter() - t0

    def _infer(self, texts: list):
        t0 = time.perf_counter()
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)
        self.infer_total_ms = (time.perf_counter() - t0) * 1000.0
        self.infer_avg_ms = self.infer_total_ms / max(len(texts), 1)
        return preds

    def evaluate(self, texts: list, labels: list) -> dict:
        """在给定数据上评估，返回预测结果与完整指标（含混淆矩阵）。"""
        preds = self._infer(texts)
        classes = self.clf.classes_.tolist()
        return {
            "preds": preds.tolist(),
            "classes": classes,
            "confusion_matrix": confusion_matrix(labels, preds, labels=classes).tolist(),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, preds),
        }

    def predict(self, texts: list) -> list:
        """批量预测（单独调用入口），并记录推理耗时。"""
        return self._infer(texts).tolist()
