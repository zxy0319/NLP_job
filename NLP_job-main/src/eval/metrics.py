# -*- coding: utf-8 -*-
"""评测指标计算模块（成员 C 负责）。

实现二分类/多分类通用的：
- 混淆矩阵（Confusion Matrix）
- 每类 Precision / Recall / F1 与支持数（Support）
- Macro-Precision / Macro-Recall / Macro-F1（对各类取平均，缓解类别不均衡）
- Accuracy（总体正确率）

口径说明（与组内规范一致）：
- 本任务标签空间为 {"positive", "negative"} 二分类；
- Macro-F1 = 两个类别 F1 的算术平均，而非按样本数加权；
- 预测标签必须在标准标签空间内，非法标签单独统计为 INVALID。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Sequence


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """返回 cm[true][pred] = count 的嵌套字典。

    labels 未提供时，由 y_true/y_pred 的并集自动确定（排序保证稳定输出）。
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        # 预测标签非法时归入 INVALID 桶，避免 KeyError 中断评测
        t = t if t in cm else None
        p = p if p in labels else "INVALID"
        if t is None:
            continue
        cm[t][p] = cm[t].get(p, 0) + 1
    return cm


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def per_class_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """返回每类指标，形如 {label: {tp, fp, fn, support, precision, recall, f1}}。"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels)
    report = {}
    for c in labels:
        tp = cm[c].get(c, 0)
        fp = sum(cm[t].get(c, 0) for t in labels if t != c)
        fn = sum(cm[c].get(p, 0) for p in labels if p != c)
        support = tp + fn  # 真实标签为 c 的样本数
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        report[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return report


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """汇总指标：accuracy、macro P/R/F1、per_class 明细与混淆矩阵。"""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    report = per_class_report(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels)

    n = len(y_true)
    correct = sum(cm[c].get(c, 0) for c in labels)
    accuracy = _safe_div(correct, n)

    macro_precision = _safe_div(sum(r["precision"] for r in report.values()), len(labels))
    macro_recall = _safe_div(sum(r["recall"] for r in report.values()), len(labels))
    macro_f1 = _safe_div(sum(r["f1"] for r in report.values()), len(labels))

    return {
        "n": n,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": report,
        "confusion_matrix": cm,
        "labels": labels,
    }
