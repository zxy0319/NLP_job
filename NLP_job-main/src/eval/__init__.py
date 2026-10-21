# -*- coding: utf-8 -*-
"""成员 C 负责的统一评测模块。

职责：
- 读取 outputs/ 下各方案的预测文件与 data/processed/test.jsonl 真实标签；
- 计算 Precision / Recall / Macro-F1 / Accuracy 与混淆矩阵；
- 落盘 logs/eval_*.log 作为工作量证据；
- 自动生成 Baseline vs LLM 横向对比表。
"""
