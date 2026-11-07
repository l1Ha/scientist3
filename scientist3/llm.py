"""LLM backends.

Scientist3 runs end-to-end with ``scripted`` agents and zero network calls.  A
live OpenAI-compatible backend can be selected with ``--llm openai`` and is
documented in ``docs/live-models.md``.  Creative agents are the only consumers
of this module: every promotion, rejection and score comparison is computed
from trusted experiment observations, never from a model response.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    mode: str = "scripted"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls, mode: str = "scripted", model: str = "") -> "LLMConfig":
        return cls(
            mode=mode,
            model=model or os.environ.get("SCIENTIST3_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get(
                "SCIENTIST3_BASE_URL", "https://api.openai.com/v1"
            ),
            api_key=os.environ.get("SCIENTIST3_API_KEY", ""),
        )


class ScriptedLLM:
    """Deterministic backend used by the offline pipeline.

    Creative responses are produced by the calling agent from case metadata and
    trusted observations.  This backend exists so that tests and the default
    demo never require network access or API keys.
    """

    mode = "scripted"

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        raise LLMError(
            "scripted agents do not call complete(); use the deterministic "
            "agent responses or run with --llm openai"
        )


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat-completions client (stdlib only)."""

    mode = "openai"

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise LLMError(
                "live mode needs an API key: set SCIENTIST3_API_KEY "
                "(or pass --api-key)"
            )
        self.config = config

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM request failed: {exc.reason}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response: {body!r}") from exc


def build_llm(config: LLMConfig):
    if config.mode in {"scripted", "offline", "none"}:
        return ScriptedLLM()
    if config.mode in {"openai", "openai-compatible", "live"}:
        return OpenAICompatLLM(config)
    if config.mode in {"anthropic", "gemini"}:
        raise LLMError(
            f"{config.mode} is not wired directly; point --base-url at an "
            "OpenAI-compatible gateway instead"
        )
    raise LLMError(f"unknown LLM mode: {config.mode}")


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response."""

    text = text.strip()
    candidates = [text]
    candidates.extend(_JSON_BLOCK.findall(text))
    if "[" in text and "]" in text:
        candidates.append(text[text.find("[") : text.rfind("]") + 1])
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"could not parse JSON from model response: {text[:300]!r}")
