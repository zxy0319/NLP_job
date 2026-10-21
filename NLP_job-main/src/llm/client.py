"""标准库 HTTP 客户端；Key 只读环境变量，不输出请求头或服务端错误正文。"""
from dataclasses import dataclass
from http.client import HTTPException
import json
import logging
import math
import os
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .prompt_templates import parse_label


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 64
    timeout: float = 60.0
    max_retries: int = 2
    retry_delay: float = 1.0
    input_price: float | None = None
    output_price: float | None = None
    currency: str = "CNY"
    seed: int = 42
    send_seed: bool = False
    thinking: str | None = None
    cache_input_price: float | None = None

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("base URL 不能包含凭据、查询参数或 fragment")
        if not url.hostname or (url.scheme != "https" and not (
                url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost", "::1"})):
            raise ValueError("远程 API 必须使用 HTTPS；HTTP 仅允许本机")
        if not self.model.strip():
            raise ValueError("必须通过 --model 或 LLM_MODEL 指定实际模型名")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_tokens <= 0 or self.max_retries < 0:
            raise ValueError("timeout/max_tokens 必须为正数；max_retries 不能为负")
        if not math.isfinite(self.retry_delay) or self.retry_delay < 0:
            raise ValueError("retry_delay 必须为有限非负数")
        prices = (self.input_price, self.output_price)
        if (prices[0] is None) != (prices[1] is None):
            raise ValueError("输入和输出单价必须同时配置，单位为每百万 Token")
        if any(p is not None and (not math.isfinite(p) or p < 0) for p in prices):
            raise ValueError("单价必须为有限非负数")
        if self.thinking not in {None, "enabled", "disabled"}:
            raise ValueError("thinking 必须为 enabled 或 disabled")
        if self.cache_input_price is not None and (
                self.input_price is None or not math.isfinite(self.cache_input_price)
                or not 0 <= self.cache_input_price <= self.input_price):
            raise ValueError("缓存输入单价必须与输入/输出单价一起配置，且介于 0 和输入单价之间")


USAGE_COUNTERS = ("http_attempts", "prompt_tokens_known", "completion_tokens_known", "usage_missing_attempts",
                  "prompt_cache_hit_tokens_known", "prompt_cache_miss_tokens_known", "cache_usage_missing_attempts")


def usage_cost(counts: dict, config: ClientConfig) -> dict:
    """缺少缓存用量时按普通输入单价估上界，同时明确标记不完整。"""
    has_usage = counts["http_attempts"] > counts["usage_missing_attempts"]
    cost = None
    if config.input_price is not None and has_usage:
        cost = (counts["prompt_tokens_known"]*config.input_price +
                counts["completion_tokens_known"]*config.output_price)
        if config.cache_input_price is not None:
            cost -= counts["prompt_cache_hit_tokens_known"]*(config.input_price-config.cache_input_price)
        cost /= 1_000_000
    complete = (config.input_price is not None and counts["usage_missing_attempts"] == 0 and
                (config.cache_input_price is None or counts["cache_usage_missing_attempts"] == 0))
    return {"known_cost": cost, "cost_complete": complete, "currency": config.currency}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # 避免 Authorization 被自动转发到另一服务。


class LLMClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self._key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self._key:
            raise ValueError("缺少 API Key：请设置 LLM_API_KEY 或 OPENAI_API_KEY 环境变量")
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self.logger = logging.getLogger("llm")
        self.stats = {key: 0 for key in USAGE_COUNTERS}
        self.stats["http_latency_ms_total"] = 0.0

    def _add_stats(self, **values):
        with self._stats_lock:
            for key, value in values.items():
                self.stats[key] += value

    def complete(self, messages: list) -> dict:
        config = self.config
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens, "stream": False}
        if config.send_seed:
            payload["seed"] = config.seed
        if config.thinking is not None:
            payload["thinking"] = {"type": config.thinking}
        if not hasattr(self._local, "opener"):
            self._local.opener = build_opener(NoRedirect())
        url = config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        start = time.perf_counter()
        local = {key: 0 for key in USAGE_COUNTERS}
        for attempt in range(config.max_retries + 1):
            usage_known = False
            missing_recorded = False
            local["http_attempts"] += 1
            self._add_stats(http_attempts=1)
            t0 = time.perf_counter()
            try:
                request = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
                with self._local.opener.open(request, timeout=config.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("响应必须是 JSON 对象")
                usage = data.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ValueError("usage 必须是 JSON 对象")
                p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
                usage_known = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in (p, c))
                if usage_known:
                    local["prompt_tokens_known"] += p
                    local["completion_tokens_known"] += c
                    self._add_stats(prompt_tokens_known=p, completion_tokens_known=c)
                    hit, miss = usage.get("prompt_cache_hit_tokens"), usage.get("prompt_cache_miss_tokens")
                    if all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in (hit, miss)) and hit+miss == p:
                        local["prompt_cache_hit_tokens_known"] += hit
                        local["prompt_cache_miss_tokens_known"] += miss
                        self._add_stats(prompt_cache_hit_tokens_known=hit, prompt_cache_miss_tokens_known=miss)
                    elif config.cache_input_price is not None:
                        local["cache_usage_missing_attempts"] += 1
                        self._add_stats(cache_usage_missing_attempts=1)
                choice = data["choices"][0]
                if not isinstance(choice, dict):
                    raise ValueError("choice 必须是 JSON 对象")
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    raise ValueError("回答被截断或过滤；请检查 max-tokens 或模型配置")
                label = parse_label(choice["message"]["content"])
                self.logger.info("API attempt=%d latency_ms=%.3f prompt_tokens=%s completion_tokens=%s usage_known=%s", attempt+1,
                                 (time.perf_counter()-t0)*1000, p if usage_known else None,
                                 c if usage_known else None, usage_known)
                if not usage_known:
                    local["usage_missing_attempts"] += 1
                    self._add_stats(usage_missing_attempts=1)
                    missing_recorded = True
                return {"pred_label": label, "latency_ms": round((time.perf_counter()-start)*1000, 3),
                        "prompt_tokens": local["prompt_tokens_known"] if usage_known else None,
                        "completion_tokens": local["completion_tokens_known"] if usage_known else None,
                        "response_model": data.get("model", config.model), **usage_cost(local, config), **local}
            except HTTPError as exc:
                code = exc.code
                exc.close()
                error = f"HTTP {code}"
                retryable = code in {408, 429} or 500 <= code < 600
            except (URLError, OSError, HTTPException, TimeoutError, socket.timeout):
                error, retryable = "连接失败、响应传输中断或超时", True
            except (ValueError, KeyError, IndexError, TypeError):
                error, retryable = "响应结构、结束原因或情感标签无效", True
            finally:
                self._add_stats(http_latency_ms_total=(time.perf_counter()-t0)*1000)
                if not usage_known and not missing_recorded:
                    local["usage_missing_attempts"] += 1
                    self._add_stats(usage_missing_attempts=1)
            self.logger.warning("API attempt=%d failed=%s latency_ms=%.3f", attempt+1, error,
                                (time.perf_counter()-t0)*1000)
            if not retryable or attempt == config.max_retries:
                raise RuntimeError(f"模型调用失败：{error}；已尝试 {attempt+1} 次。已完成的样本可断点续跑") from None
            time.sleep(min(config.retry_delay * 2**attempt, 30))
        raise RuntimeError("无法完成调用")
