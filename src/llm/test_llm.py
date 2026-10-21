"""B 模块回归测试：python -m unittest src.llm.test_llm -v。

HTTP 服务仅模拟协议，所有测试结果写入临时目录。
"""
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


AVAILABLE = importlib.util.find_spec("src.llm") is not None
if AVAILABLE:
    from src.llm.client import ClientConfig, LLMClient
    from src.llm.prompt_templates import build_messages, parse_label, split_dev
    from src.llm.run_llm import load_data, run_experiment


class AvailabilityTests(unittest.TestCase):
    def test_b_module_exists(self):
        self.assertTrue(AVAILABLE, "B 的大模型流水线尚未实现")


@unittest.skipUnless(AVAILABLE, "等待 B 模块实现")
class PromptTests(unittest.TestCase):
    def test_extracts_unambiguous_labels(self):
        for text, expected in [("positive", "positive"), ("```json\n{\"label\":\"negative\"}\n```", "negative"),
                               ("情感标签：正面", "positive"), ("结论：negative。", "negative")]:
            with self.subTest(text=text):
                self.assertEqual(parse_label(text), expected)

    def test_rejects_ambiguous_or_missing_labels(self):
        for text in ["positive or negative", "not positive", "并非正面", "neutral", "", "<think>positive</think>",
                     "This is not a positive review.", "不是很正面", "不积极"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_label(text)

    def test_only_text_goes_into_query(self):
        messages = build_messages("隐藏标签样本", [])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("隐藏标签样本", messages[-1]["content"])
        self.assertNotIn("gold_label", messages[-1]["content"])

    def test_dev_examples_have_both_classes_and_are_excluded(self):
        rows = [{"text": str(i), "label": "positive" if i % 2 else "negative"} for i in range(20)]
        examples, validation = split_dev(rows, 3, 42, 8)
        self.assertEqual(len(examples), 3)
        self.assertEqual({r["label"] for r in examples}, {"positive", "negative"})
        self.assertFalse({r["text"] for r in examples} & {r["text"] for r in validation})
        self.assertEqual((examples, validation), split_dev(rows, 3, 42, 8))


class ProtocolHandler(BaseHTTPRequestHandler):
    calls = []
    status_sequence = []
    omit_usage = False
    response_override = None
    truncate_response = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append((self.path, payload))
        status = type(self).status_sequence.pop(0) if type(self).status_sequence else 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if type(self).truncate_response:
            self.send_header("Content-Length", "1000")
        self.end_headers()
        if type(self).truncate_response:
            self.wfile.write(b'{"choices":')
            return
        if status != 200:
            self.wfile.write(b'{"error":"do not log response secrets"}')
            return
        query = payload["messages"][-1]["content"]
        label = "positive" if "满意" in query else "negative"
        result = {"choices": [{"message": {"content": label}, "finish_reason": "stop"}]}
        if not type(self).omit_usage:
            result["usage"] = {"prompt_tokens": 10, "completion_tokens": 2,
                               "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 2}
        if type(self).response_override is not None:
            result = type(self).response_override
        self.wfile.write(json.dumps(result).encode())


@unittest.skipUnless(AVAILABLE, "等待 B 模块实现")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ProtocolHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        ProtocolHandler.calls = []
        ProtocolHandler.status_sequence = []
        ProtocolHandler.omit_usage = False
        ProtocolHandler.response_override = None
        ProtocolHandler.truncate_response = False

    def client(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "unit-test-only"}):
            return LLMClient(ClientConfig(base_url=self.url, model="protocol-test", retry_delay=0,
                                          input_price=2, output_price=4))

    def test_retries_and_counts_unknown_billing(self):
        ProtocolHandler.status_sequence = [429, 200]
        client = self.client()
        result = client.complete(build_messages("满意", []))
        self.assertEqual(result["pred_label"], "positive")
        self.assertEqual(result["http_attempts"], 2)
        self.assertEqual(result["usage_missing_attempts"], 1)
        self.assertAlmostEqual(result["known_cost"], 0.000028)
        self.assertEqual(ProtocolHandler.calls[0][0], "/v1/chat/completions")

    def test_auth_errors_are_not_retried_and_key_is_not_logged(self):
        ProtocolHandler.status_sequence = [401]
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as cm:
            self.client().complete(build_messages("满意", []))
        self.assertEqual(len(ProtocolHandler.calls), 1)
        self.assertNotIn("unit-test-only", str(cm.exception))

    def test_missing_usage_is_unknown_not_free(self):
        ProtocolHandler.omit_usage = True
        result = self.client().complete(build_messages("满意", []))
        self.assertIsNone(result["prompt_tokens"])
        self.assertIsNone(result["known_cost"])
        self.assertEqual(result["usage_missing_attempts"], 1)

    def test_deepseek_non_thinking_and_cache_pricing(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "unit-test-only"}):
            client = LLMClient(ClientConfig(base_url=self.url, model="protocol-test",
                                           thinking="disabled", input_price=2, output_price=8,
                                           cache_input_price=0.04))
        result = client.complete(build_messages("满意", []))
        self.assertEqual(ProtocolHandler.calls[0][1]["thinking"], {"type": "disabled"})
        self.assertEqual(result["prompt_cache_hit_tokens_known"], 8)
        self.assertEqual(result["prompt_cache_miss_tokens_known"], 2)
        self.assertAlmostEqual(result["known_cost"], 0.00002032)

    def test_missing_or_invalid_cache_counts_are_not_claimed_complete(self):
        for usage in ({"prompt_tokens": 10, "completion_tokens": 2},
                      {"prompt_tokens": 10, "completion_tokens": 2,
                       "prompt_cache_hit_tokens": 100, "prompt_cache_miss_tokens": 0}):
            with self.subTest(usage=usage):
                ProtocolHandler.response_override = {
                    "choices": [{"message": {"content": "positive"}, "finish_reason": "stop"}], "usage": usage}
                with patch.dict(os.environ, {"LLM_API_KEY": "unit-test-only"}):
                    client = LLMClient(ClientConfig(base_url=self.url, model="protocol-test",
                                                   input_price=2, output_price=8, cache_input_price=0.04))
                result = client.complete(build_messages("满意", []))
                self.assertFalse(result["cost_complete"])
                self.assertEqual(result["cache_usage_missing_attempts"], 1)
                self.assertAlmostEqual(result["known_cost"], 0.000036)

    def test_parallel_predictions_keep_order_and_exact_usage_on_resume(self):
        rows = [{"text": f"{'满意' if i % 2 else '很糟'}-{i}",
                 "label": "positive" if i % 2 else "negative"} for i in range(25)]
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d)/"parallel.jsonl"
            client = self.client()
            result = run_experiment(rows, [], client, checkpoint, logging.getLogger("test"), workers=4)
            self.assertEqual([p["text"] for p in result["predictions"]], [r["text"] for r in rows])
            self.assertEqual(result["metrics"]["accuracy"], 1)
            self.assertEqual(result["usage"]["http_attempts"], 25)
            self.assertEqual(client.stats["prompt_tokens_known"], 250)
            recovered = run_experiment(rows, [], client, checkpoint, logging.getLogger("test"), workers=3)
            self.assertEqual(recovered["new_http_attempts"], 0)
            self.assertEqual(recovered["usage"], result["usage"])

    def test_malformed_api_structure_is_retried_with_safe_error(self):
        ProtocolHandler.response_override = []
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, "已尝试 3 次"):
            client.complete(build_messages("满意", []))
        self.assertEqual(client.stats["http_attempts"], 3)
        self.assertEqual(client.stats["usage_missing_attempts"], 3)

    def test_missing_key_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "缺少 API Key"):
            LLMClient(ClientConfig(base_url=self.url, model="protocol-test"))

    def test_truncated_http_response_is_retried(self):
        ProtocolHandler.truncate_response = True
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, "已尝试 3 次"):
            client.complete(build_messages("满意", []))
        self.assertEqual(client.stats["http_attempts"], 3)
        self.assertEqual(client.stats["usage_missing_attempts"], 3)

    def test_resume_rejects_changed_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"cache.jsonl"
            rows = [{"text": "满意", "label": "positive"}]
            run_experiment(rows, [], self.client(), p, logging.getLogger("test"))
            with self.assertRaisesRegex(ValueError, "缓存与当前数据或配置不一致"):
                run_experiment([{"text": "很糟", "label": "negative"}], [], self.client(), p, logging.getLogger("test"))

    def test_resume_recovers_interrupted_last_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"cache.jsonl"
            rows = [{"text": "满意", "label": "positive"}]
            client = self.client()
            run_experiment(rows, [], client, p, logging.getLogger("test"))
            with p.open("ab") as stream:
                stream.write(b'{"identity":')
            recovered = run_experiment(rows, [], client, p, logging.getLogger("test"))
            self.assertEqual(recovered["new_http_attempts"], 0)
            self.assertEqual(recovered["metrics"]["accuracy"], 1)

    def test_resume_reuses_completed_rows_and_preserves_schema(self):
        rows = [{"text": "满意", "label": "positive"}, {"text": "很糟", "label": "negative"}]
        with tempfile.TemporaryDirectory() as d:
            client = self.client()
            a = run_experiment(rows, [], client, Path(d)/"cache.jsonl", logging.getLogger("test"))
            b = run_experiment(rows, [], client, Path(d)/"cache.jsonl", logging.getLogger("test"))
            self.assertEqual(a["metrics"]["accuracy"], 1)
            self.assertEqual(b["new_http_attempts"], 0)
            self.assertEqual(len(ProtocolHandler.calls), 2)
            self.assertEqual(set(b["predictions"][0]), {"text", "gold_label", "pred_label", "latency_ms"})

    def test_cli_end_to_end_selects_on_dev_and_does_not_modify_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dev = root/"dev.jsonl"
            test = root/"test.jsonl"
            dev.write_text("\n".join(json.dumps({"text": f"{'满意' if i%2 else '很糟'}开发{i}",
                                                     "label": "positive" if i%2 else "negative"}) for i in range(10)), encoding="utf-8")
            test.write_text(json.dumps({"text": "满意测试", "label": "positive"})+"\n"+
                            json.dumps({"text": "很糟测试", "label": "negative"})+"\n", encoding="utf-8")
            before = test.read_bytes()
            env = dict(os.environ, LLM_API_KEY="unit-test-only")
            command = [sys.executable, "-m", "src.llm.run_llm", "--base-url", self.url, "--model", "protocol-test",
                       "--dev", str(dev), "--test", str(test), "--output-dir", str(root/"out"),
                       "--log-dir", str(root/"logs"), "--dev-limit", "4", "--input-price", "2", "--output-price", "4"]
            result = subprocess.run(command, env=env, text=True, capture_output=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((root/"out"/"llm_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selection_split"], "dev")
            self.assertEqual(summary["selected_strategy"], "zeroshot")
            self.assertEqual(summary["current_process"]["http_attempts"], 12)
            self.assertEqual(summary["experiment_totals"]["logical_predictions"], 12)
            self.assertEqual(summary["experiment_totals"]["prompt_tokens_known"], 120)
            self.assertEqual(summary["experiment_totals"]["completion_tokens_known"], 24)
            self.assertAlmostEqual(summary["experiment_totals"]["known_cost"], 0.000336)
            self.assertEqual(len((root/"out"/"preds_llm.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(before, test.read_bytes())
            logs = "".join(p.read_text(encoding="utf-8") for p in (root/"logs").glob("*.log"))
            self.assertNotIn("unit-test-only", logs)

    def test_invalid_data_fails_before_network(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.jsonl"
            p.write_text('{"text":"", "label":"positive"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_data(p)


if __name__ == "__main__":
    unittest.main()
