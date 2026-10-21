# -*- coding: utf-8 -*-
"""数据清洗与三划分脚本。

功能（对应任务书要求）：
1. 读取 data/raw/dataset.jsonl 原始数据，兼容常见开源数据集字段
   （text/sentence/content/review 等；label/label_desc/sentiment 等，
   数值标签可通过 LABEL_MAPPING 映射为语义标签）；
2. 清洗（去空、去重、非法 JSON 跳过），并对金融/个人敏感信息脱敏；
3. 按 label 分层抽样（Stratified Split）切分：每个类别内打乱后按 8:1:1
   抽样合并，保证类别不均衡时 dev/test 中各类占比与全局一致；
   采用最大余数法分配名额，总量仍精确为 80%/10%/10%；
4. 全程输出带时间戳的日志（随机种子、样本条数、字段长度统计），
   同步保存到 logs/data_process_YYYYMMDD.log 作为工作量证据。
"""

import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
RAW_DATA_PATH = Path("data/raw/dataset.jsonl")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs")

# 字段适配：常见开源中文数据集（THUCNews/ChnSentiCorp/CLUENER 等）
# 原始字段名可能不同，按候选顺序取第一个非空字段
TEXT_FIELDS = ["text", "sentence", "content", "review", "comment"]
LABEL_FIELDS = ["label", "label_desc", "sentiment", "class", "category"]
# 数值标签映射：如原始标签为 0/1/2，可配置例如
# {"0": "negative", "1": "positive", "2": "neutral"}；为空则保留原始标签值
LABEL_MAPPING = {}

# 敏感信息脱敏规则：身份证(18位) -> 手机号(11位) -> 银行卡(16-19位) -> 邮箱
SENSITIVE_PATTERNS = [
    ("ID_CARD", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("BANK_CARD", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"data_process_{datetime.now():%Y%m%d}.log"
    logger = logging.getLogger("data_process")
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


def mask_sensitive(text: str) -> str:
    for name, pattern in SENSITIVE_PATTERNS:
        text = pattern.sub(f"[MASKED_{name}]", text)
    return text


def pick_field(item: dict, candidates: list):
    for key in candidates:
        if key in item and str(item[key]).strip():
            return key, str(item[key]).strip()
    return None, None


def load_and_clean(logger: logging.Logger) -> list:
    samples, seen = [], set()
    used_fields = None
    with RAW_DATA_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipped invalid JSON at line {line_no}")
                continue
            text_field, text = pick_field(item, TEXT_FIELDS)
            label_field, label = pick_field(item, LABEL_FIELDS)
            if text_field is None or label_field is None:
                continue
            if used_fields is None:
                used_fields = (text_field, label_field)
            label = LABEL_MAPPING.get(label, label)
            if not text or not label:
                continue
            if text in seen:
                continue
            seen.add(text)
            samples.append({"text": mask_sensitive(text), "label": label})
    if used_fields:
        logger.info(
            f"Field mapping applied: text <- '{used_fields[0]}', label <- '{used_fields[1]}'"
        )
    return samples


def split_samples(samples: list) -> dict:
    rng = random.Random(RANDOM_SEED)
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    n = len(samples)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_dev = int(n * SPLIT_RATIOS["dev"])

    def allocate(target: int) -> dict:
        raw = {lb: len(by_label[lb]) / n * target for lb in sorted(by_label)}
        base = {lb: int(v) for lb, v in raw.items()}
        rest = target - sum(base.values())
        for lb in sorted(by_label, key=lambda k: raw[k] - base[k], reverse=True)[:rest]:
            base[lb] += 1
        return base

    q_train = allocate(n_train)
    q_dev = allocate(n_dev)
    q_test = {lb: len(by_label[lb]) - q_train[lb] - q_dev[lb] for lb in by_label}

    splits = {"train": [], "dev": [], "test": []}
    for lb in sorted(by_label):
        group = by_label[lb][:]
        rng.shuffle(group)
        a, b = q_train[lb], q_train[lb] + q_dev[lb]
        splits["train"].extend(group[:a])
        splits["dev"].extend(group[a:b])
        splits["test"].extend(group[b:])
    return splits


def avg_text_len(samples: list) -> float:
    if not samples:
        return 0.0
    return sum(len(s["text"]) for s in samples) / len(samples)


def main():
    logger = setup_logger()
    logger.info(f"Initialized data processing with random seed: {RANDOM_SEED}")
    logger.info(f"Reading raw data from: {RAW_DATA_PATH.as_posix()}")
    samples = load_and_clean(logger)
    logger.info(f"Total valid samples after cleaning: {len(samples)}")

    splits = split_samples(samples)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        out_path = PROCESSED_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(
            f"Split [{name}.jsonl] written: {len(data)} samples "
            f"| Avg text len: {avg_text_len(data):.2f}"
        )

    logger.info("Data pipeline finished successfully. Files saved to data/processed")


if __name__ == "__main__":
    main()
