"""Minimal client for an OpenAI-compatible chat API (DeepSeek), with retries and an on-disk response cache.

The key comes from $DEEPSEEK_API_KEY or ~/.config/deepseek/api_key and is never written to the repository,
logs or the cache. Cache entries are keyed by the full request body, so reruns cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    return key.strip() if key else Path("~/.config/deepseek/api_key").expanduser().read_text().strip()


class ChatClient:
    def __init__(self, model: str, cache_path: str | Path, **defaults) -> None:
        self.model, self.defaults, self.key = model, defaults, api_key()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]
                except (json.JSONDecodeError, KeyError):
                    continue
        self.lock = threading.Lock()

    def chat(self, messages: list[dict], **params) -> dict:
        """{'content': str, 'usage': dict} for one chat completion (cached)."""
        body = {"model": self.model, "messages": messages, **self.defaults, **params}
        key = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        data = json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "User-Agent": "savemv-pilot/0.1"}
        for attempt in range(7):
            try:
                req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data, headers=headers)
                resp = json.load(urllib.request.urlopen(req, timeout=600))
                out = {"content": resp["choices"][0]["message"].get("content") or "", "usage": resp.get("usage", {})}
                break
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 402, 403, 422):
                    raise RuntimeError(f"chat request rejected: HTTP {e.code} {e.read()[:200]!r}") from None
                time.sleep(min(60, 2 ** attempt))
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError("chat request failed after retries")
        with self.lock:
            self.cache[key] = out
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": out}) + "\n")
        return out


class LocalReader(ChatClient):
    """Same interface, answered by a local Hugging Face chat model (greedy, one request at a time on the GPU)."""

    def __init__(self, path: str, cache_path: str | Path, device: str = "cuda:0") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model, self.defaults, self.key = path, {}, ""
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]
                except (json.JSONDecodeError, KeyError):
                    continue
        self.lock, self.gpu = threading.Lock(), threading.Lock()
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def chat(self, messages: list[dict], max_tokens: int = 256, **_) -> dict:
        key = hashlib.sha1(json.dumps({"reader": self.model, "messages": messages, "max_tokens": max_tokens},
                                      sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with self.gpu, self.torch.inference_mode():
            gen = self.lm.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        out = {"content": self.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip(),
               "usage": {"prompt_tokens": int(ids.shape[1]), "completion_tokens": int(gen.shape[1] - ids.shape[1])}}
        with self.lock:
            self.cache[key] = out
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": out}) + "\n")
        return out
