"""Lightweight chat LLM client (no heavy deps).

Design goals for this repo:
- Keep dependencies minimal (stdlib only).
- Support OpenAI-compatible HTTP APIs when available.
- Always allow a safe fallback path when LLM is disabled/unavailable.

Default env prefix is ``RESUME_LLM_`` (same as resume text LLM). Callers that need
other backends pass ``env_prefix=`` (e.g. ``JOB_LLM_``, ``RESUME_VLLM_``).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    # Ensure .env is loaded even if import order differs.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatLLMConfig:
    provider: str
    model: str
    api_key: Optional[str]
    base_url: str
    timeout_seconds: int
    temperature: float
    max_tokens: int

    @staticmethod
    def from_env(prefix: str = "RESUME_LLM_") -> "ChatLLMConfig":
        provider = (os.getenv(f"{prefix}PROVIDER") or "debug").strip().lower()
        model = (os.getenv(f"{prefix}MODEL") or "gpt-4.1-mini").strip()
        # Only accept project-scoped key name to avoid ambiguity.
        api_key = (os.getenv(f"{prefix}API_KEY") or "").strip() or None
        base_url = (os.getenv(f"{prefix}BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
        timeout_seconds = int(os.getenv(f"{prefix}TIMEOUT_SECONDS") or "30")
        temperature = float(os.getenv(f"{prefix}TEMPERATURE") or "0.2")
        max_tokens = int(os.getenv(f"{prefix}MAX_TOKENS") or "700")
        return ChatLLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class ChatLLMClient:
    """Chat LLM client that can return structured JSON.

    Providers:
    - debug: always raises NotImplementedError for network call; caller should fallback
    - openai_compatible: calls POST {base_url}/chat/completions
    """

    def __init__(self, config: Optional[ChatLLMConfig] = None, *, env_prefix: str = "RESUME_LLM_") -> None:
        self._env_prefix = env_prefix
        self._config: Optional[ChatLLMConfig] = config

    @property
    def config(self) -> ChatLLMConfig:
        # Always refresh from env to avoid stale config when env/.env changes
        # or when import order loads dotenv after this module.
        self._config = ChatLLMConfig.from_env(self._env_prefix)
        return self._config

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        cfg = self.config
        if cfg.provider == "debug":
            raise LLMClientError("LLM provider is 'debug' (disabled).")
        if cfg.provider in ("openai", "openai_compatible", "compatible"):
            return self._openai_compatible_json(system_prompt=system_prompt, user_prompt=user_prompt)
        raise LLMClientError(f"Unsupported LLM provider: {cfg.provider}")

    async def generate_json_with_images(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images_png_base64: List[str],
    ) -> Dict[str, Any]:
        """OpenAI-compatible vision: user message with text + data:image/png;base64,... parts."""
        cfg = self.config
        if cfg.provider == "debug":
            raise LLMClientError("LLM provider is 'debug' (disabled).")
        if cfg.provider not in ("openai", "openai_compatible", "compatible"):
            raise LLMClientError(f"Unsupported LLM provider: {cfg.provider}")
        parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for b64 in images_png_base64:
            b64 = (b64 or "").strip()
            if not b64:
                continue
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        if len(parts) <= 1:
            raise LLMClientError("generate_json_with_images requires at least one image")
        return self._openai_compatible_chat(
            system_prompt=system_prompt,
            user_content=parts,
        )

    def _openai_compatible_json(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self._openai_compatible_chat(
            system_prompt=system_prompt,
            user_content=user_prompt,
        )

    def _openai_compatible_chat(
        self,
        *,
        system_prompt: str,
        user_content: Union[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        cfg = self.config
        key_name = f"{self._env_prefix}API_KEY"
        if not cfg.api_key:
            if key_name in os.environ and not (os.environ.get(key_name) or "").strip():
                raise LLMClientError(f"{key_name} is set but empty in this process environment")
            raise LLMClientError(f"Missing {key_name}")

        url = f"{cfg.base_url}/chat/completions"
        payload = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {cfg.api_key}")

        started = time.time()
        try:
            with urlopen(req, timeout=cfg.timeout_seconds) as r:
                raw = r.read().decode("utf-8")
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise LLMClientError(f"LLM HTTP {e.code}: {body}") from e
        except URLError as e:
            raise LLMClientError(f"LLM network error: {e}") from e
        finally:
            _ = started

        try:
            resp = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            raise LLMClientError(f"LLM response not JSON: {raw[:500]}") from e

        content = (
            (((resp.get("choices") or [{}])[0] or {}).get("message") or {}).get("content")
            if isinstance(resp, dict)
            else None
        )
        if not content or not isinstance(content, str):
            raise LLMClientError(f"LLM response missing content: {str(resp)[:500]}")

        return _parse_json_from_text(content)


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """Best-effort parse JSON object from LLM text."""
    s = (text or "").strip()
    # Fast path: direct JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try extract the largest {...} block
    m = _JSON_BLOCK_RE.search(s)
    if not m:
        raise LLMClientError(f"LLM output is not JSON: {s[:500]}")
    try:
        obj = json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        raise LLMClientError(f"Failed to parse JSON from LLM output: {s[:500]}") from e
    if not isinstance(obj, dict):
        raise LLMClientError("LLM output JSON is not an object")
    return obj

