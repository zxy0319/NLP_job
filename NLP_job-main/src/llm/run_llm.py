"""B 成员运行入口：开发集选提示、冻结测试集比较、统计成本和断点续跑。"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import compute_metrics
from src.llm.client import ClientConfig, LLMClient
from src.llm.prompt_templates import LABELS, PROMPT_VERSION, build_messages, split_dev

ROOT = Path(__file__).resolve().parents[2]


def load_data(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError("text 必须为非空字符串")
                if row.get("label") not in LABELS:
                    raise ValueError("label 必须是 positive 或 negative")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name} 第 {n} 行格式无效") from exc
            rows.append({"text": row["text"], "label": row["label"]})
    if not rows:
        raise ValueError(f"{path.name} 没有有效样本")
    return rows


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value, jsonl: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    temp.replace(path)


def run_experiment(rows: list, examples: list, client: LLMClient, checkpoint: Path,
                   logger: logging.Logger) -> dict:
    """checkpoint 带数据、提示与配置指纹；恢复已完成请求，不复用不相符缓存。"""
    identity = digest({"rows": rows, "examples": examples, "config": asdict(client.config), "prompt": PROMPT_VERSION})
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    if checkpoint.exists():
        # 恢复写入中断的末行；中间行损坏明确报错。
        raw = checkpoint.read_bytes()
        lines = raw.splitlines(keepends=True)
        good_bytes = 0
        for i, line in enumerate(lines):
            if i == len(lines)-1 and not line.endswith(b"\n"):
                logger.warning("恢复被中断的 checkpoint 末行")
                with checkpoint.open("r+b") as stream:
                    stream.truncate(good_bytes)
                break
            item = json.loads(line)
            if item["identity"] != identity or item["index"] != len(completed):
                raise ValueError("缓存与当前数据或配置不一致；请使用新的 output-dir")
            if item["result"]["pred_label"] not in LABELS:
                raise ValueError("checkpoint 含非法标签")
            completed.append(item["result"])
            good_bytes += len(line)
        if len(completed) > len(rows):
            raise ValueError("checkpoint 样本数超出输入")
    cached = len(completed)
    new_start = client.stats["http_attempts"]
    logger.info("实验 samples=%d cached=%d shots=%d", len(rows), cached, len(examples))
    with checkpoint.open("a", encoding="utf-8", newline="\n") as stream:
        for i in range(cached, len(rows)):
            # API 只接收 text；gold 标签仅用于本地评测。
            result = client.complete(build_messages(rows[i]["text"], examples))
            stream.write(json.dumps({"identity": identity, "index": i, "result": result}, ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed.append(result)
            if (i+1) % 10 == 0 or i+1 == len(rows):
                logger.info("进度 %d/%d", i+1, len(rows))
    predictions = [{"text": row["text"], "gold_label": row["label"],
                    "pred_label": result["pred_label"], "latency_ms": result["latency_ms"]}
                   for row, result in zip(rows, completed)]
    stats = {key: sum(r[key] for r in completed) for key in
             ("http_attempts", "prompt_tokens_known", "completion_tokens_known", "usage_missing_attempts")}
    known_costs = [r["known_cost"] for r in completed if r["known_cost"] is not None]
    stats.update({"known_cost": sum(known_costs) if known_costs else None,
                  "cost_complete": client.config.input_price is not None and stats["usage_missing_attempts"] == 0,
                  "currency": client.config.currency})
    return {"metrics": compute_metrics([r["label"] for r in rows], [r["pred_label"] for r in completed], LABELS),
            "predictions": predictions, "samples": len(rows), "cached_samples": cached,
            "new_http_attempts": client.stats["http_attempts"]-new_start,
            "mean_latency_ms": sum(r["latency_ms"] for r in completed)/len(rows), "usage": stats}


def comparison(results: dict) -> str:
    lines = ["| Split | Strategy | N | Accuracy | Macro-F1 | Mean latency ms | Prompt tokens known | Completion tokens known | Cost known |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for split, strategies in results.items():
        for strategy, result in strategies.items():
            m, u = result["metrics"], result["usage"]
            cost = "未配置/未知" if u["known_cost"] is None else f"{u['known_cost']:.8f} {u['currency']}"
            if not u["cost_complete"]:
                cost += "（不完整）"
            lines.append(f"| {split} | {strategy} | {result['samples']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                         f"{result['mean_latency_ms']:.3f} | {u['prompt_tokens_known']} | {u['completion_tokens_known']} | {cost} |")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"), help="含 /v1 的 API 根地址，或完整 chat/completions 地址")
    p.add_argument("--model", default=os.getenv("LLM_MODEL"), help="服务商实际支持的模型名")
    p.add_argument("--dev", type=Path, default=ROOT/"data/processed/dev.jsonl")
    p.add_argument("--test", type=Path, default=ROOT/"data/processed/test.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    p.add_argument("--log-dir", type=Path, default=ROOT/"logs")
    p.add_argument("--strategy", choices=["compare", "zeroshot", "fewshot"], default="compare")
    p.add_argument("--shots", type=int, default=3)
    p.add_argument("--dev-limit", type=int, default=100, help="开发集比较样本数；0 表示除示例外全部")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--send-seed", action="store_true", help="仅在服务商支持 seed 参数时启用")
    p.add_argument("--input-price", type=float, help="每百万输入 Token 价格；不含缓存折扣等")
    p.add_argument("--output-price", type=float, help="每百万输出 Token 价格")
    p.add_argument("--currency", choices=["CNY", "USD"], default="CNY")
    p.add_argument("--preview", action="store_true", help="不联网，只验证数据和生成调用规模清单，不生成预测")
    return p


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("llm")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    for handler in [logging.FileHandler(args.log_dir/f"llm_run_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
                    logging.StreamHandler()]:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    start = time.perf_counter()
    client = None
    try:
        logger.info("mode=%s seed=%d prompt=%s", "preview_no_inference" if args.preview else "api", args.seed, PROMPT_VERSION)
        if args.dev.resolve() == args.test.resolve():
            raise ValueError("dev 与 test 不能是同一文件")
        dev_hash = hashlib.sha256(args.dev.read_bytes()).hexdigest()
        test_hash = hashlib.sha256(args.test.read_bytes()).hexdigest()
        dev, test = load_data(args.dev), load_data(args.test)
        if {r["text"] for r in dev} & {r["text"] for r in test}:
            raise ValueError("dev 与 test 存在重复文本，请由数据负责人检查；流水线不会修改源数据")
        examples, validation = split_dev(dev, args.shots, args.seed, args.dev_limit)
        strategies = ["zeroshot", "fewshot"] if args.strategy == "compare" else [args.strategy]
        names = ["llm_input_check.json", "llm_summary.json", "llm_comparison.md", "preds_llm.jsonl"]
        names += [f"preds_llm_{s}.jsonl" for s in strategies]
        if any((args.output_dir/name).resolve() in {args.dev.resolve(), args.test.resolve()} for name in names):
            raise ValueError("输出文件与数据源路径冲突")
        input_check = {"status": "preview_no_inference" if args.preview else "validated",
                       "seed": args.seed, "dev_sha256": dev_hash, "test_sha256": test_hash,
                       "dev_samples": len(dev), "test_samples": len(test),
                       "test_label_counts": dict(Counter(r["label"] for r in test)),
                       "fewshot_examples": examples, "dev_validation_samples": len(validation),
                       "planned_logical_calls": (len(test)+len(validation))*len(strategies),
                       "max_http_attempts": (len(test)+len(validation))*len(strategies)*(args.max_retries+1),
                       "selection_split": "dev", "predictions_generated": False}
        logger.info("数据只读校验 dev=%d test=%d dev_sha256=%s test_sha256=%s", len(dev), len(test), dev_hash, test_hash)
        if args.preview:
            atomic_json(args.output_dir/"llm_input_check.json", input_check)
            logger.info("离线检查完成：计划调用=%d（不含重试）；本次 API 调用=0；未生成预测或实验指标", input_check["planned_logical_calls"])
            return 0
        if not args.base_url or not args.model:
            raise ValueError("请先设置 LLM_BASE_URL 和 LLM_MODEL，或使用 --base-url/--model；无账号可运行 --preview")
        config = ClientConfig(base_url=args.base_url, model=args.model, temperature=args.temperature,
                              max_tokens=args.max_tokens, timeout=args.timeout, max_retries=args.max_retries,
                              input_price=args.input_price, output_price=args.output_price, currency=args.currency,
                              seed=args.seed, send_seed=args.send_seed)
        client = LLMClient(config)
        logger.info("模型/API配置（不含 Key）：%s", json.dumps(asdict(config), ensure_ascii=False))
        if config.input_price is None:
            logger.warning("未配置单价，成本显示为未知；完整成本报告必须同时提供 --input-price 和 --output-price")
        # 一个配置一个目录；数据或模板变化也会生成新指纹。
        run_id = digest({"config": asdict(config), "dev": dev_hash, "test": test_hash,
                         "examples": examples, "validation": validation, "prompt": PROMPT_VERSION})[:20]
        cache_dir = args.output_dir/"llm_checkpoints"/run_id
        results = {"dev": {}, "test": {}}
        for strategy in strategies:
            logger.info("开发集 strategy=%s", strategy)
            results["dev"][strategy] = run_experiment(validation, examples if strategy == "fewshot" else [], client,
                                                     cache_dir/f"dev_{strategy}.jsonl", logger)
        # 只用 dev 的 Macro-F1 选主方案；同分比较 Accuracy，再同分选 zeroshot 节约提示成本。
        selected = max(strategies, key=lambda s: (results["dev"][s]["metrics"]["macro_f1"],
                                                  results["dev"][s]["metrics"]["accuracy"], s == "zeroshot"))
        logger.info("开发集确定主方案=%s；接下来冻结提示，运行测试集", selected)
        for strategy in strategies:
            logger.info("测试集 strategy=%s", strategy)
            results["test"][strategy] = run_experiment(test, examples if strategy == "fewshot" else [], client,
                                                      cache_dir/f"test_{strategy}.jsonl", logger)
        if hashlib.sha256(args.test.read_bytes()).hexdigest() != test_hash or hashlib.sha256(args.dev.read_bytes()).hexdigest() != dev_hash:
            raise ValueError("实验期间源数据发生变化，拒绝发布结果")
        for strategy in strategies:
            atomic_json(args.output_dir/f"preds_llm_{strategy}.jsonl", results["test"][strategy]["predictions"], jsonl=True)
        atomic_json(args.output_dir/"preds_llm.jsonl", results["test"][selected]["predictions"], jsonl=True)
        table = comparison(results)
        for split in results.values():
            for result in split.values():
                del result["predictions"]
        process = dict(client.stats)
        process["mean_http_latency_ms"] = process["http_latency_ms_total"]/process["http_attempts"] if process["http_attempts"] else None
        process["known_cost"] = None if config.input_price is None else (
            process["prompt_tokens_known"]*config.input_price + process["completion_tokens_known"]*config.output_price)/1_000_000
        process["cost_complete"] = config.input_price is not None and process["usage_missing_attempts"] == 0
        summary = {"status": "completed", "run_id": run_id, "selection_split": "dev", "selected_strategy": selected,
                   "input_check": input_check, "config": asdict(config), "results": results,
                   "current_process": process, "wall_seconds": time.perf_counter()-start,
                   "cost_note": "每百万输入/输出Token单价线性估算；未知usage的调用、缓存折扣及其他计费项以服务商账单为准。"}
        summary["input_check"]["predictions_generated"] = True
        atomic_json(args.output_dir/"llm_summary.json", summary)
        report = args.output_dir/"llm_comparison.md"
        report.write_text(f"# B 成员大模型实验结果\n\n主方案：{selected}，根据开发集选定。\n\n{table}\n\n"
                          f"本次进程费用统计：`{json.dumps(process, ensure_ascii=False)}`\n\n"
                          f"墙钟时间：{summary['wall_seconds']:.3f} 秒。\n\n{summary['cost_note']}\n", encoding="utf-8")
        logger.info("\n%s", table)
        logger.info("当前进程统计=%s total_seconds=%.3f；已发布标准预测文件", json.dumps(process), summary["wall_seconds"])
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        logger.error("运行未完成：%s", exc)
        if client:
            logger.error("本次进程已发生调用（包括失败调用）：%s", json.dumps(client.stats))
        return 1
    finally:
        logger.info("进程结束 wall_seconds=%.3f", time.perf_counter()-start)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
