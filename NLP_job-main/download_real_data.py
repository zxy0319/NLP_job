# -*- coding: utf-8 -*-
"""读取本地下载的 TSV 真实数据集并转换为 data/raw/dataset.jsonl"""

import json
from pathlib import Path

def main():
    raw_dir = Path("data/raw")
    out_file = raw_dir / "dataset.jsonl"
    tsv_files = ["train.tsv", "dev.tsv", "test.tsv"]

    total = 0
    with out_file.open("w", encoding="utf-8") as f_out:
        for fname in tsv_files:
            file_path = raw_dir / fname
            if not file_path.exists():
                print(f"[提示] 未找到 {fname}，已跳过")
                continue
            
            with file_path.open("r", encoding="utf-8") as f_in:
                lines = f_in.readlines()
                # 判断是否有表头
                start = 1 if lines and ("label" in lines[0] or "text" in lines[0]) else 0
                for line in lines[start:]:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        label_col, text_col = parts[0].strip(), parts[1].strip()
                        # 自动纠正列顺序
                        if label_col in ["0", "1"]:
                            lbl, txt = label_col, text_col
                        elif text_col in ["0", "1"]:
                            lbl, txt = text_col, label_col
                        else:
                            continue
                        
                        if not txt:
                            continue
                        label_str = "positive" if lbl == "1" else "negative"
                        f_out.write(json.dumps({"text": txt, "label": label_str}, ensure_ascii=False) + "\n")
                        total += 1

    print(f"[SUCCESS] 真实公开数据集转换完毕！共生成 {total} 条数据至: {out_file.as_posix()}")

if __name__ == "__main__":
    main()