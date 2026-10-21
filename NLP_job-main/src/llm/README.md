# B 成员大模型情感分类

在现有项目根目录执行以下命令。B 模块使用 Python 3.10 及以上版本的标准库，复用 `src/eval/metrics.py`；无需新增安装依赖。

## 模块职责

- `client.py`：兼容 OpenAI 的 Chat Completions 接口、环境变量密钥、超时重试、Token 和耗时记录。
- `prompt_templates.py`：Zero-shot、Few-shot 提示模板，开发集示例抽样及严格标签解析。
- `run_llm.py`：开发集对比选方案、测试集批量推理、断点恢复、标准预测文件及费用统计。
- `test_llm.py`：本机 HTTP 协议和流水线回归测试，数据和结果均使用临时目录。

## 配置和运行

密钥只从 `LLM_API_KEY` 或 `OPENAI_API_KEY` 读取，不写入代码。服务地址使用 `LLM_BASE_URL` 或 `OPENAI_BASE_URL`，模型名使用 `LLM_MODEL`；程序不自动读取 `.env`。模型应支持 `messages`、`temperature` 和 `max_tokens` 参数；服务地址可填写 API 根地址或完整的 `/chat/completions` 地址。

PowerShell：

```powershell
$secureKey = Read-Host 'API Key' -AsSecureString
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
$env:LLM_BASE_URL = Read-Host 'API 根地址，例如服务商文档中的 https 地址'
$env:LLM_MODEL = Read-Host '实际模型名'
$inputPrice = Read-Host '每百万输入 Token 单价'
$outputPrice = Read-Host '每百万输出 Token 单价'
python -m src.llm.run_llm --strategy compare --input-price $inputPrice --output-price $outputPrice --currency CNY
```

Linux 或 macOS 的 Bash：

```bash
read -r -s -p 'API Key: ' LLM_API_KEY
echo
export LLM_API_KEY
read -r -p 'API 根地址: ' LLM_BASE_URL
export LLM_BASE_URL
read -r -p '实际模型名: ' LLM_MODEL
export LLM_MODEL
read -r -p '每百万输入 Token 单价: ' input_price
read -r -p '每百万输出 Token 单价: ' output_price
python -m src.llm.run_llm --strategy compare --input-price "$input_price" --output-price "$output_price" --currency CNY
```

币种按实际账单选择 `CNY` 或 `USD`。费用估算为 `(输入 Token × 输入单价 + 输出 Token × 输出单价) / 1000000`，不含缓存折扣、免费额度和税费。未配置单价时费用显示未知；缺少服务端 usage 时只累计已知用量，并将 `cost_complete` 标记为 false。

不需要账号的检查命令：

```bash
python -m unittest src.llm.test_llm -v
python -m src.llm.run_llm --preview
python -m src.llm.run_llm --help
```

`--preview` 不调用 API，不生成预测或模型指标；只在 `outputs/llm_input_check.json` 和日志中记录数据检查与计划调用规模。当前尚未配置真实 API 账号，真实实验结果需接入服务后运行生成。

## 参数对比与数据隔离

默认 seed 为 42、temperature 为 0。Few-shot 从开发集抽 3 条示例，包含正负两类，开发验证集排除示例及同文记录；API 只接收待预测文本，不接收其真实标签。先在相同的 100 条开发验证样本上比较 Zero-shot 与 Few-shot，按 Macro-F1 选主方案，同分比较 Accuracy，再同分选择 Zero-shot。然后固定提示，在冻结测试集上运行两种方案，测试指标不用于调参或选择主方案。

默认数据为 `data/processed/dev.jsonl` 和 `data/processed/test.jsonl`，路径从项目根目录解析。当前开发集 990 条、测试集 992 条，默认共计划 2184 次逻辑请求，不含重试。`--dev-limit 0` 使用排除示例后的完整开发集；`--shots` 调整示例数；`--strategy zeroshot` 或 `fewshot` 可单独运行，任务要求的对比应使用默认 `compare`。

原始数据只读，dev/test 存在重复文本或字段非法时会报错。无法解析的模型回答会重试，仍失败则停止并保留断点，不猜测标签。

## 与组员对接的结果

| 路径 | 内容 |
| --- | --- |
| `outputs/preds_llm.jsonl` | 主方案测试预测，供成员 C 直接评测 |
| `outputs/preds_llm_zeroshot.jsonl` | Zero-shot 测试预测 |
| `outputs/preds_llm_fewshot.jsonl` | Few-shot 测试预测 |
| `outputs/llm_summary.json` | 模型和单价配置、数据哈希、开发/测试指标、调用量、Token、费用和总耗时 |
| `outputs/llm_comparison.md` | 横向对比表，可用于报告 |
| `logs/llm_run_YYYYMMDD.log` | 实际运行日志，包含种子、模型、请求和统计 |

每行预测仅包含 `text`、`gold_label`、`pred_label`、`latency_ms`，保持测试集原顺序；两个标签字段均为 `positive` 或 `negative`。主结果生成后可用现有评测模块验证：

```bash
python -m src.eval.evaluate --pred outputs/preds_llm.jsonl
```

断点保存在被 Git 忽略的 `outputs/llm_checkpoints/`，相同命令重跑会跳过已完成样本；配置、数据或提示版本改变时使用新的指纹。`results` 统计包括恢复的历史完成请求，`current_process` 只统计本次新请求。调用量含重试；`latency_ms` 包含该样本的重试退避，`mean_http_latency_ms` 是新 HTTP 尝试的平均耗时。跨多次失败进程的全部费用需结合历史日志及账单核对。

GitHub 中提交本模块源码、测试、说明和 `.gitignore` 的新增规则；真实实验完成后再提交预测、汇总、对比表及运行日志。密钥、`.env`、断点缓存和 Python 缓存不提交。
