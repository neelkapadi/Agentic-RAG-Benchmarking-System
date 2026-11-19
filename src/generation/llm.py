"""Minimal Ollama chat client (open-source LLMs served locally)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b",
                 temperature: float = 0.0, num_ctx: int = 8192, timeout_s: int = 180,
                 max_tokens: int = 768, json_max_tokens: int = 512):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        # Capping output matters: some models in JSON mode can emit whitespace forever.
        self.json_max_tokens = json_max_tokens

    def chat(self, messages: list[dict], json_mode: bool = False,
             temperature: float | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.json_max_tokens if json_mode else self.max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        start = time.perf_counter()
        data = self._post_with_retry(payload)
        return LLMResponse(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - start,
        )

    def _post_with_retry(self, payload: dict, retries: int = 3) -> dict:
        """Ollama occasionally returns 500 (e.g. JSON-grammar hiccups) or drops a connection; retry."""
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                err: Exception = requests.HTTPError(f"{r.status_code} from Ollama: {r.text[:200]}", response=r)
            except (requests.ConnectionError, requests.Timeout) as e:
                err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
        raise err

    def chat_json(self, messages: list[dict]) -> tuple[dict, LLMResponse]:
        try:
            resp = self.chat(messages, json_mode=True)
        except requests.HTTPError:
            # Fall back to free-form output; parse_json extracts the first {...} block.
            resp = self.chat(messages, json_mode=False)
        return parse_json(resp.text), resp

    def check(self) -> None:
        """Raise a helpful error if Ollama is not running or the model is not pulled."""
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=5).json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama is not reachable at {self.base_url}. Start it with `ollama serve`.") from e
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise RuntimeError(f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}")


def parse_json(text: str) -> dict:
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1", "y"}
