"""离线核对 B 报告证据，不调用 API，不改动原始数据、日志或预测。

在项目根目录运行：python docs/audit_day4_B.py
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import compute_metrics
from src.llm.prompt_templates import LABELS, split_dev


def main():
    out = ROOT / "outputs/day4_B"
    out.mkdir(parents=True, exist_ok=True)
    sources = [*sorted((ROOT / "src").rglob("*.py")),
               *sorted((ROOT / "data/processed").glob("*.jsonl")),
               *sorted((ROOT / "outputs").glob("preds_*.jsonl")),
               ROOT / "logs/llm_run_20261021.log",
               ROOT / "docs/report_draft_A.md", ROOT / "docs/report_draft_shtu.md"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    read = lambda name: [json.loads(x) for x in (ROOT / name).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    dev, test = read("data/processed/dev.jsonl"), read("data/processed/test.jsonl")
    examples, validation = split_dev(dev, shots=3, seed=42, limit=100)
    assert len(examples) == 3 and len(validation) == 100
    assert not ({r['text'] for r in examples} & {r['text'] for r in validation})
    assert not ({r['text'] for r in dev} & {r['text'] for r in test})
    lines = (ROOT / "logs/llm_run_20261021.log").read_text(encoding="utf-8").splitlines()
    config_line = next(x for x in lines if "模型/API配置" in x)
    config = json.loads(config_line[config_line.index('{'):])
    totals_line = next(x for x in lines if "全部实验统计（包含断点恢复）=" in x)
    recorded_total = json.loads(totals_line.split("=", 1)[1])
    stages, stage, calls = [], None, []
    for n, line in enumerate(lines, 1):
        match = re.search(r"(开发集|测试集) strategy=(zeroshot|fewshot)", line)
        if match:
            stage = {"split": "dev" if match[1] == "开发集" else "test", "strategy": match[2],
                     "log_start_line": n, "calls": []}
            stages.append(stage)
        match = re.search(r"API attempt=(\d+) latency_ms=([\d.]+) prompt_tokens=(\d+) completion_tokens=(\d+) usage_known=True", line)
        if match:
            item = {"attempt": int(match[1]), "http_latency_ms": float(match[2]),
                    "prompt_tokens": int(match[3]), "completion_tokens": int(match[4]), "line": n}
            assert stage is not None
            stage["calls"].append(item)
            calls.append(item)
    assert len(stages) == 4 and len(calls) == 2184
    assert all(c["attempt"] == 1 for c in calls)
    assert not any("[WARNING]" in x or "[ERROR]" in x for x in lines)
    ledger = []
    for s in stages:
        records = s.pop("calls")
        s["samples"] = len(records)
        s["prompt_tokens"] = sum(c["prompt_tokens"] for c in records)
        s["completion_tokens"] = sum(c["completion_tokens"] for c in records)
        s["total_tokens"] = s["prompt_tokens"] + s["completion_tokens"]
        s["log_last_api_line"] = records[-1]["line"]
        cost = (Decimal(s["prompt_tokens"]) * Decimal(str(config["input_price"])) +
                Decimal(s["completion_tokens"]) * Decimal(str(config["output_price"]))) / Decimal(1000000)
        s["estimated_cost_CNY"] = str(cost)
        s["mean_cost_per_prediction_CNY"] = str(cost / s["samples"])
        row = next(x for x in lines if x.startswith(f"| {s['split']} | {s['strategy']} |"))
        cells = [x.strip() for x in row.strip('|').split('|')]
        assert int(cells[2]) == s["samples"]
        assert int(cells[6]) == s["prompt_tokens"] and int(cells[7]) == s["completion_tokens"]
        assert Decimal(cells[8].split()[0]) == cost
        s["accuracy_log_rounded"] = float(cells[3])
        s["macro_f1_log_rounded"] = float(cells[4])
        s["mean_latency_ms_log_rounded"] = float(cells[5])
        ledger.append(dict(s))
    assert sum(s['prompt_tokens'] for s in stages) == recorded_total['prompt_tokens_known']
    assert sum(s['completion_tokens'] for s in stages) == recorded_total['completion_tokens_known']
    assert sum(Decimal(s['estimated_cost_CNY']) for s in stages) == Decimal(str(recorded_total['known_cost']))
    predictions = {}
    for name in ("zeroshot", "fewshot"):
        rows = read(f"outputs/preds_llm_{name}.jsonl")
        assert len(rows) == len(test) == 992
        assert all(r['text'] == t['text'] and r['gold_label'] == t['label'] and r['pred_label'] in LABELS
                   for r, t in zip(rows, test))
        m = compute_metrics([r['label'] for r in test], [r['pred_label'] for r in rows], LABELS)
        lat = sorted(r['latency_ms'] for r in rows)
        assert all(math.isfinite(x) and x >= 0 for x in lat)
        s = next(s for s in stages if s['split'] == 'test' and s['strategy'] == name)
        assert round(m['accuracy'], 4) == s['accuracy_log_rounded']
        assert round(m['macro_f1'], 4) == s['macro_f1_log_rounded']
        assert round(statistics.mean(lat), 3) == s['mean_latency_ms_log_rounded']
        predictions[name] = {"metrics": m, "legal_labels": len(rows),
                             "latency_ms": {"mean": statistics.mean(lat), "median": statistics.median(lat),
                                            "p95_nearest_rank": lat[math.ceil(.95 * len(lat)) - 1],
                                            "min": lat[0], "max": lat[-1]}}
    assert (ROOT/'outputs/preds_llm.jsonl').read_bytes() == (ROOT/'outputs/preds_llm_fewshot.jsonl').read_bytes()
    assert hashes['data/processed/dev.jsonl'] == 'e83b760ebc4dd7a31368533bb18fa3ef2728c67df2cb92c040c967cbb7272e13'
    assert hashes['data/processed/test.jsonl'] == 'a75f906b15fb49698b253a6596efd3907e07e10572cf7c36bbd22bd1a13dd9a7'
    for p in sources:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.relative_to(ROOT).as_posix()]
    evidence = {"status": "passed", "api_calls_during_audit": 0, "historical_config": config,
                "stages": stages, "recorded_experiment_totals": recorded_total,
                "predictions_recomputed": predictions, "fewshot_examples_reconstructed": examples,
                "dev_metrics_note": "仅能核对历史日志；原始 dev 预测和 checkpoint 未随项目提供，不能独立重算开发集指标。",
                "format_note": "原始回答未保存；合法标签率为解析后的输出契约符合率。",
                "source_sha256": hashes}
    (out/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (out/'token_cost_ledger.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    print('PASS: 4 experiment stages; 2184 logged calls; token/cost totals match; 2 x 992 predictions aligned.')
    print('PASS: test metrics and latency match; main predictions equal fewshot; frozen data hashes match.')
    print('PASS: original source/data/log/prediction/report files unchanged; audit API calls = 0.')
    print('Written: outputs/day4_B/evidence.json and token_cost_ledger.csv')


if __name__ == '__main__':
    main()
