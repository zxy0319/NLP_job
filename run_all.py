# -*- coding: utf-8 -*-
"""项目一键复现脚本（成员 C）。

一条命令串起「数据校验 → 基线 → 原理点对比 → (可选)LLM → 评测对比 → 坏例归因」
的完整验证流程，避免老师复现时手动逐条敲命令。

用法（在仓库根目录 NLP_job/ 下执行）：
    python run_all.py                  # 默认：校验 + 基线 + 评测 + 坏例（LLM 有 Key 才跑）
    python run_all.py --skip-baseline  # 复用已提交的预测文件，只重跑评测 / 坏例（最快）
    python run_all.py --with-llm       # 强制运行 LLM 推理（需先配置 LLM_API_KEY 等环境变量）
    python run_all.py --re-split       # 额外重新执行数据划分（默认跳过，避免覆盖冻结数据）

依赖安装：pip install -r requirements.txt
说明：LLM 步骤需要 API Key（环境变量 LLM_API_KEY / OPENAI_API_KEY），未配置时自动跳过，
      并复用仓库已提交的 outputs/preds_llm.jsonl 完成横向对比。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def has_llm_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def run(step: str, script: str, *args: str) -> bool:
    """以项目根目录为 cwd 运行单个脚本步骤，返回是否成功。"""
    cmd = [PY, str(ROOT / script), *args]
    print("\n" + "=" * 68)
    print(f"[{step}] 运行：{' '.join(cmd)}")
    print("=" * 68)
    try:
        proc = subprocess.run(cmd, cwd=ROOT)
    except Exception as exc:  # noqa: BLE001
        print(f"[{step}] 异常退出：{exc}")
        return False
    ok = proc.returncode == 0
    print(f"[{step}] {'成功' if ok else f'失败（退出码 {proc.returncode}）'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="一键复现：数据→基线→评测→坏例")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过基线/原理点重训，复用已提交的预测文件")
    parser.add_argument("--with-llm", action="store_true",
                        help="强制运行 LLM 推理（需配置 API 环境变量）")
    parser.add_argument("--re-split", action="store_true",
                        help="额外重新执行数据划分（默认跳过，避免覆盖冻结数据）")
    args = parser.parse_args()

    results: list[tuple[str, bool]] = []

    # 1) 数据（默认只校验，不重新划分）
    if args.re_split:
        results.append(("数据划分", run("数据划分", "src/data/clean_and_split.py")))
    results.append(("数据校验", run("数据校验", "src/data/check_data.py")))

    # 2) 非大模型基线 + 原理点 2
    if not args.skip_baseline:
        results.append(("基线训练+预测", run("基线训练+预测", "src/baseline/run_baseline.py")))
        results.append(("原理点2(Word2Vec)", run("原理点2(Word2Vec)", "src/baseline/exp_principle2.py")))

    # 3) 大模型方案（有 Key 或显式 --with-llm 才跑）
    if args.with_llm or has_llm_key():
        if not has_llm_key():
            print("\n[警告] 已指定 --with-llm 但未检测到 LLM_API_KEY / OPENAI_API_KEY，LLM 步骤可能失败。")
        results.append(("LLM 推理", run("LLM 推理", "src/llm/run_llm.py", "--strategy", "compare")))
    else:
        print("\n[LLM] 未检测到 API Key，跳过 LLM 推理，复用已提交的 outputs/preds_llm.jsonl。")

    # 4) 统一评测：横向对比 + 全指标表（outputs/comparison.md）
    results.append(("横向对比", run("横向对比", "src/eval/evaluate.py", "--compare")))

    # 5) 坏例归因
    results.append(("基线坏例归因", run("基线坏例归因", "src/eval/analyze_bad_cases.py",
                                        "--pred", "outputs/preds_baseline.jsonl")))
    results.append(("LLM 坏例归因", run("LLM 坏例归因", "src/eval/analyze_bad_cases.py",
                                        "--pred", "outputs/preds_llm.jsonl", "--model", "llm")))

    # 汇总
    print("\n" + "=" * 68)
    print("一键复现结果汇总")
    print("=" * 68)
    for name, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n共 {len(results)} 步，成功 {n_ok} 步。")
    if n_ok != len(results):
        print("存在失败步骤，请查看对应 logs/*.log 定位问题。")
        return 1
    print("全部通过，复现完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
