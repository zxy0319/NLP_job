# B 任务交付说明

本交付完成中文情感二分类的大模型推理、Zero-shot 与 3-shot 对比、费用与延迟统计，以及真实运行日志。所有预测和统计由 2026 年 10 月 21 日调用 DeepSeek 官方 API 运行生成；协议测试结果没有混入正式实验。

## 实验结果

模型请求名为 `deepseek-flash`，使用非思考模式，temperature=0。本地抽样 seed=42；Few-shot 的 3 条示例来自开发集，包含正负两类。先在排除示例后的相同 100 条开发验证样本上选择主方案，再固定提示对 992 条冻结测试样本运行两种方案。

| 数据集 | 方案 | 样本数 | Accuracy | Macro-F1 | 平均响应延迟 ms | 费用估算 元 |
| --- | --- | --- | --- | --- | --- | --- |
| 开发验证集 | Zero-shot | 100 | 0.9200 | 0.9197 | 818.418 | 0.032752 |
| 开发验证集 | Few-shot 3 | 100 | 0.9400 | 0.9398 | 840.461 | 0.065152 |
| 冻结测试集 | Zero-shot | 992 | 0.9083 | 0.9082 | 755.967 | 0.310036 |
| 冻结测试集 | Few-shot 3 | 992 | 0.9204 | 0.9202 | 770.894 | 0.631444 |

主方案为 Few-shot，根据开发集 Macro-F1 选定。`outputs/preds_llm.jsonl` 与 `outputs/preds_llm_fewshot.jsonl` 内容相同；测试集指标只用于最终评测。

全部实验共 2,184 次 HTTP 调用，未发生重试；Prompt Token 为 510,956，Completion Token 为 2,184。接口报告缓存命中 0 Token、未命中 510,956 Token。按本次高峰时段输入未命中 2 元/百万、缓存命中 0.04 元/百万、输出 8 元/百万计算，费用估算合计 **1.039384 元**。总运行耗时约 283.795 秒，全部逻辑请求平均响应延迟约 769.475 ms。单次响应延迟与并发墙钟总耗时是不同统计口径。

费用是按真实 API usage 与运行时段单价估算，实际扣费以服务商账单为准。[官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。

## 应合入团队仓库的文件

| 路径 | 用途 |
| --- | --- |
| `src/llm/__init__.py` | 模块入口标记 |
| `src/llm/client.py` | 环境变量凭据、超时重试、并发安全统计、DeepSeek 非思考模式及缓存计费 |
| `src/llm/prompt_templates.py` | 提示模板、3-shot 抽样、标签规整 |
| `src/llm/run_llm.py` | 开发集选方案、全测试集推理、顺序输出、断点与汇总 |
| `src/llm/test_llm.py` | 本机协议与端到端回归测试 |
| `src/llm/README.md` | PowerShell/Linux 配置、运行命令、单价说明 |
| `outputs/preds_llm.jsonl` | 主方案标准预测，992 行 |
| `outputs/preds_llm_zeroshot.jsonl` | Zero-shot 测试预测，992 行 |
| `outputs/preds_llm_fewshot.jsonl` | Few-shot 测试预测，992 行 |
| `outputs/llm_summary.json` | 配置、模型、分组指标、Token、费用与耗时 |
| `outputs/llm_comparison.md` | 可用于报告的横向对比表 |
| `outputs/llm_input_check.json` | 数据哈希、示例、调用计划与推理状态 |
| `outputs/llm_verification.json` | 文件格式、样本顺序、数据不变、指标与费用核对结果 |
| `logs/llm_run_20261021.log` | 本次完整 API 运行日志及统计 |
| `logs/eval_20261021.log` | 使用成员 C 评测入口验证主方案的日志 |
| `B_SUBMISSION.md` | 本说明 |

每行正式预测严格只有 `text`、`gold_label`、`pred_label`、`latency_ms` 四个字段。三份文件均逐行对应测试集原始顺序，标签仅为 `positive` 或 `negative`。

## 验证证据

项目测试发现命令共运行 19 项测试，全部通过：

```bash
python -m unittest discover -s src -p 'test*.py' -v
```

成员 C 的现有评测命令成功读取主方案全部 992 条预测，复算 Accuracy=0.9204、Macro-F1=0.9202：

```bash
python -m src.eval.evaluate --pred outputs/preds_llm.jsonl
```

另已独立逐行核对三份预测的字段、顺序、真实标签、有限非负延迟，并独立复算 Accuracy/Macro-F1、检查主方案选择与费用总和。训练集、开发集、测试集的运行前后 SHA-256 均一致，完整哈希见 `outputs/llm_verification.json`。源码、预测、说明和运行日志已扫描确认不含 API 密钥。

本机 Codex 随附 Python 3.12 完成以上运行与验证。系统的 `python/py` 尚无可用安装；组员复现时需在自己的 Python 3.10 及以上环境执行 README 命令。B 模块无需第三方依赖，复用项目既有的 `src/eval/metrics.py`。可选出图依赖 matplotlib，本次环境未安装，因此交付中不包含新混淆矩阵图；数值评测正常完成。

## 提交方法

`B_submission_20261021.zip` 是合入团队现有仓库的增量文件包，内部直接是 `src/`、`outputs/`、`logs/` 和本说明。它不重复打包其他成员的代码和冻结数据。把包内文件按相同相对路径合入团队项目根目录，避免在仓库里再次创建一层 `NLP_job-main/`。

使用克隆的团队仓库建立自己的分支，只暂存上表中的文件，再检查提交内容。完整目录名和运行缓存不要上传。尤其确认 `.gitignore` 中保留 `.env`、`__pycache__/` 和 `outputs/llm_checkpoints/` 的忽略规则。交付包排除了密钥、环境配置和断点缓存。

本地当前工作目录是 ZIP 解压副本，没有 `.git`，此次未自动创建 Git 提交或上传 GitHub。团队仓库若同时已有其他人的修改，先比较文件差异，再合并 B 模块对应内容。
