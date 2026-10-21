# B 成员大模型实验结果

主方案：fewshot，根据开发集选定。

| Split | Strategy | N | Accuracy | Macro-F1 | Mean latency ms | Prompt tokens known | Completion tokens known | Cost known |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | zeroshot | 100 | 0.9200 | 0.9197 | 818.418 | 15976 | 100 | 0.03275200 CNY |
| dev | fewshot | 100 | 0.9400 | 0.9398 | 840.461 | 32176 | 100 | 0.06515200 CNY |
| test | zeroshot | 992 | 0.9083 | 0.9082 | 755.967 | 151050 | 992 | 0.31003600 CNY |
| test | fewshot | 992 | 0.9204 | 0.9202 | 770.894 | 311754 | 992 | 0.63144400 CNY |

全部实验统计（含已恢复的完成样本）：`{"http_attempts": 2184, "prompt_tokens_known": 510956, "completion_tokens_known": 2184, "usage_missing_attempts": 0, "prompt_cache_hit_tokens_known": 0, "prompt_cache_miss_tokens_known": 510956, "cache_usage_missing_attempts": 0, "known_cost": 1.039384, "cost_complete": true, "currency": "CNY", "logical_predictions": 2184, "mean_latency_ms": 769.4750389194138}`

本次进程费用统计：`{"http_attempts": 2184, "prompt_tokens_known": 510956, "completion_tokens_known": 2184, "usage_missing_attempts": 0, "prompt_cache_hit_tokens_known": 0, "prompt_cache_miss_tokens_known": 510956, "cache_usage_missing_attempts": 0, "http_latency_ms_total": 1680559.1213999996, "mean_http_latency_ms": 769.486777197802, "known_cost": 1.039384, "cost_complete": true, "currency": "CNY"}`

墙钟时间：283.795 秒。

按配置单价及API返回的输入、输出、缓存命中Token估算；单价须与运行时段相符。未知usage、历史失败进程及其他计费项以服务商账单为准。
