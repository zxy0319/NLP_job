# 横向对比（Baseline vs LLM）

## 汇总

| 方案 | Accuracy | Macro-P | Macro-R | Macro-F1 | 平均延迟(ms) | 样本数 |
| --- | --- | --- | --- | --- | --- | --- |
| preds_baseline.jsonl | 0.8730 | 0.8726 | 0.8719 | 0.8722 | 0.27 | 992 |
| preds_llm.jsonl | 0.9204 | 0.9200 | 0.9222 | 0.9202 | 770.89 | 992 |
| preds_llm_fewshot.jsonl | 0.9204 | 0.9200 | 0.9222 | 0.9202 | 770.89 | 992 |
| preds_llm_zeroshot.jsonl | 0.9083 | 0.9093 | 0.9112 | 0.9082 | 755.97 | 992 |
| preds_principle2_dense.jsonl | 0.8034 | 0.8028 | 0.8014 | 0.8020 | 0.10 | 992 |

## 全指标明细

| 方案 | Accuracy | Macro-F1 | P(positive) | P(negative) | R(positive) | R(negative) | F1(positive) | F1(negative) | 平均延迟(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| preds_baseline.jsonl | 0.8730 | 0.8722 | 0.8773 | 0.8678 | 0.8872 | 0.8565 | 0.8822 | 0.8621 | 0.27 |
| preds_llm.jsonl | 0.9204 | 0.9202 | 0.9521 | 0.8880 | 0.8966 | 0.9478 | 0.9235 | 0.9169 | 770.89 |
| preds_llm_fewshot.jsonl | 0.9204 | 0.9202 | 0.9521 | 0.8880 | 0.8966 | 0.9478 | 0.9235 | 0.9169 | 770.89 |
| preds_llm_zeroshot.jsonl | 0.9083 | 0.9082 | 0.9546 | 0.8639 | 0.8703 | 0.9522 | 0.9105 | 0.9059 | 755.97 |
| preds_principle2_dense.jsonl | 0.8034 | 0.8020 | 0.8092 | 0.7964 | 0.8289 | 0.7739 | 0.8189 | 0.7850 | 0.10 |
