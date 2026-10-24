# 中文评论情感分类（ChnSentiCorp）

《自然语言处理》课程项目。给定一条中文商品 / 酒店 / 图书评论，判断其情感极性为 `positive`（正面）或 `negative`（负面）。

技术路线为**双轨并行**：① 非大模型基线（jieba 分词 + TF-IDF + 逻辑回归）；② 大模型方案（DeepSeek `deepseek-flash` 的 Zero-shot / Few-shot 提示）；两者在同一冻结测试集上以 Macro-F1 为主指标横向对比，并做坏例归因与一键复现。

## 环境与依赖

- Python 3.10+
- 安装依赖：

```bash
pip install -r requirements.txt
```

依赖：`jieba`、`scikit-learn`、`gensim`、`numpy`、`matplotlib`（仅出图用）。

## 一键复现

在仓库根目录执行：

```bash
python run_all.py
```

默认流程：**数据校验 → 基线训练+预测 → 原理点 2（Word2Vec）→ 横向对比 → 坏例归因**。

| 参数 | 作用 |
| --- | --- |
| `--skip-baseline` | 复用已提交预测文件，只重跑评测 / 坏例（最快） |
| `--with-llm` | 强制运行 LLM 推理（需先配置 API 环境变量） |
| `--re-split` | 额外重新执行数据划分（默认跳过，避免覆盖冻结数据） |

> LLM 步骤需要 API Key；未配置时自动跳过，并复用仓库已提交的 `outputs/preds_llm.jsonl` 完成横向对比。

## 随机种子与标签

- 随机种子 `seed = 42`；标签空间 `["positive", "negative"]`。
- 主指标 **Macro-F1**（对 positive / negative 两类 F1 取算术平均，缓解类别不均衡）。

## 预期输出（冻结测试集 992 条）

| 方案 | Accuracy | Macro-F1 | 平均延迟(ms) |
| --- | --- | --- | --- |
| Baseline（TF-IDF + LR） | 0.8730 | 0.8722 | 0.27 |
| **LLM Few-shot（主方案）** | **0.9204** | **0.9202** | 770.89 |
| LLM Zero-shot | 0.9083 | 0.9082 | 755.97 |
| Word2Vec 稠密（原理点 2） | 0.8034 | 0.8020 | 0.10 |

评测与坏例结果分别写入 `outputs/comparison.md`、`outputs/bad_cases_*.json`，日志落 `logs/*.log`。

## 大模型方案（可选，需 API Key）

密钥一律从环境变量读取，不写入代码：

```bash
export LLM_API_KEY='...'           # 或 OPENAI_API_KEY
export LLM_BASE_URL='https://api.deepseek.com'
export LLM_MODEL='deepseek-flash'
python -m src.llm.run_llm --strategy compare
```

无 Key 时直接运行 `python run_all.py --skip-baseline` 即可复现评测与坏例部分。LLM 详细说明见 `src/llm/README.md`。

## 分步命令

```bash
python src/data/check_data.py                          # 数据校验
python src/baseline/run_baseline.py                    # 基线训练 + 预测
python src/baseline/exp_principle2.py                  # 原理点 2（TF-IDF vs Word2Vec）
python src/eval/evaluate.py --compare                  # 横向对比（生成 outputs/comparison.md）
python src/eval/evaluate.py --pred outputs/preds_baseline.jsonl --plot   # 单方案评测 + 混淆矩阵图
python src/eval/analyze_bad_cases.py --pred outputs/preds_baseline.jsonl
python src/eval/analyze_bad_cases.py --pred outputs/preds_llm.jsonl --model llm
```

## 数据

公开数据集 ChnSentiCorp 中文情感评论，经清洗、脱敏与 8:1:1 分层划分：训练 **7924** / 开发 **990** / 测试 **992** 条。测试集全程冻结（只读、不打乱、不参与调参）。原始数据见 `data/raw/`（不入库，`download_real_data.py` 可重新下载）。

## 目录结构

```
├── run_all.py              # 一键复现
├── requirements.txt
├── data/processed/         # train/dev/test.jsonl（冻结）
├── src/
│   ├── data/               # 清洗、划分、校验
│   ├── baseline/           # TF-IDF+LR 基线、原理点 2
│   ├── llm/                # 大模型调用、提示模板、推理
│   └── eval/               # 指标计算、统一评测、坏例归因、可视化
├── outputs/                # 预测文件、对比表、坏例 JSON、混淆矩阵图
├── logs/                   # 全部运行日志（含种子/参数/耗时）
└── docs/                   # 报告草稿与最终整合报告
```
