# -*- coding: utf-8 -*-
"""数据校验脚本。

校验 data/processed 下的三划分结果：
1. 文件存在、JSONL 格式合法、字段完整（text/label）；
2. 三个集合之间无样本重叠（防数据泄漏）；
3. 敏感信息脱敏无残留；
4. 输出样本规模、平均文本长度、标签分布与各集合类别占比偏差；
5. 校验结果同步写入 logs/data_check_YYYYMMDD.log，作为报告引用的持久化证据。
"""

import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED_DIR = Path("data/processed")
EXPECTED_FILES = ["train.jsonl", "dev.jsonl", "test.jsonl"]
LOG_DIR = Path("logs")
STRATIFICATION_TOLERANCE = 0.02

SENSITIVE_RESIDUALS = [
    ("ID_CARD", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("BANK_CARD", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"data_check_{datetime.now():%Y%m%d}.log"
    logger = logging.getLogger("data_check")
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


def load_jsonl(path: Path) -> list:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.as_posix()} line {line_no}: invalid JSON ({e})")
            if "text" not in item or "label" not in item:
                raise ValueError(f"{path.as_posix()} line {line_no}: missing required field")
            if not str(item["text"]).strip():
                raise ValueError(f"{path.as_posix()} line {line_no}: empty text")
            items.append(item)
    return items


def main():
    logger = setup_logger()
    logger.info(f"Initialized data check on: {PROCESSED_DIR.as_posix()}")

    all_texts, all_labels = {}, {}
    ok = True

    for name in EXPECTED_FILES:
        path = PROCESSED_DIR / name
        if not path.exists():
            logger.error(f"Missing file: {path.as_posix()}")
            ok = False
            continue
        try:
            items = load_jsonl(path)
        except ValueError as e:
            logger.error(f"{e}")
            ok = False
            continue
        all_texts[name] = {it["text"] for it in items}
        all_labels[name] = Counter(it["label"] for it in items)

        residual = []
        for it in items:
            for pname, pattern in SENSITIVE_RESIDUALS:
                if pattern.search(it["text"]):
                    residual.append((pname, it["text"][:30]))
        if residual:
            ok = False
            logger.error(
                f"{name}: {len(residual)} samples contain unmasked sensitive info, e.g. {residual[:3]}"
            )

        labels = all_labels[name]
        avg_len = sum(len(it["text"]) for it in items) / len(items) if items else 0.0
        label_str = ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
        logger.info(
            f"[PASS] {name}: {len(items)} samples | avg_len={avg_len:.2f} | labels: {label_str}"
        )

    names = [n for n in EXPECTED_FILES if n in all_texts]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = all_texts[names[i]] & all_texts[names[j]]
            if overlap:
                ok = False
                logger.error(
                    f"Data leakage: {names[i]} and {names[j]} share {len(overlap)} identical samples"
                )
            else:
                logger.info(f"[PASS] No overlap between {names[i]} and {names[j]}")

    if len(all_labels) == len(EXPECTED_FILES):
        global_dist = sum(all_labels.values(), Counter())
        total = sum(global_dist.values())
        strat_ok = True
        for name in EXPECTED_FILES:
            n = sum(all_labels[name].values())
            for lb, cnt in sorted(global_dist.items()):
                expect = cnt / total
                actual = all_labels[name].get(lb, 0) / n if n else 0.0
                if abs(actual - expect) > STRATIFICATION_TOLERANCE:
                    strat_ok = False
                    logger.error(
                        f"Stratification drift: {name} label '{lb}' ratio {actual:.4f} "
                        f"deviates from global {expect:.4f} (>{STRATIFICATION_TOLERANCE})"
                    )
        if strat_ok:
            logger.info(
                f"[PASS] Stratification check: per-label ratios in all splits within "
                f"{STRATIFICATION_TOLERANCE:.0%} of global distribution"
            )
        else:
            ok = False

    total = sum(len(s) for s in all_texts.values())
    logger.info(f"Total samples across splits: {total}")
    logger.info("RESULT: ALL CHECKS PASSED" if ok else "RESULT: CHECKS FAILED")


if __name__ == "__main__":
    main()
