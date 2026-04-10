"""Lightweight embedding client abstraction.

支持:
- debug: 无外部依赖的哈希伪向量
- bge / sentence_transformers: 本地 HuggingFace 模型
- ollama: HTTP 调用本机 Ollama ``POST /api/embeddings``（默认 127.0.0.1:11434）
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from typing import List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import httpx


class EmbeddingClient:
    """Simple embedding client with pluggable backend.

    Interface:
    - embed_texts(texts) -> list of float vectors
    """

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        dim: int = 128,
    ) -> None:
        self.provider = (provider or os.getenv("EMBEDDING_PROVIDER") or "debug").lower()
        if model_name is not None:
            self.model_name = model_name
        elif (os.getenv("EMBEDDING_MODEL_NAME") or "").strip():
            self.model_name = (os.getenv("EMBEDDING_MODEL_NAME") or "").strip()
        elif self.provider == "ollama":
            self.model_name = "bge-m3"
        else:
            self.model_name = "BAAI/bge-small-zh-v1.5"
        self.dim = int(os.getenv("EMBEDDING_DIM") or dim)
        self._model = None
        self._ollama_base = (os.getenv("OLLAMA_EMBED_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self._ollama_timeout = float(os.getenv("OLLAMA_EMBED_TIMEOUT_SECONDS", "180"))
        # 避免 HTTP(S)_PROXY 把 127.0.0.1 交给公司代理 → 常见 502；curl 默认不走系统代理故正常。
        self._ollama_trust_env = os.getenv("OLLAMA_HTTP_TRUST_ENV", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self._ollama_truncate = os.getenv("OLLAMA_EMBED_TRUNCATE", "true").lower() not in {
            "0",
            "false",
            "no",
        }

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into float vectors.

        V1.1: 默认使用纯 Python 的 hash-based embedding，方便开发与测试。
        后续可以根据 provider 切换到 OpenAI / 本地 BGE 等真正的向量模型。
        """

        if not texts:
            return []

        if self.provider == "debug":
            return [self._hash_embed(t or "") for t in texts]

        if self.provider in {"bge", "bge_hf", "huggingface", "hf", "sentence_transformers"}:
            return await asyncio.to_thread(self._embed_texts_bge, texts)

        if self.provider == "ollama":
            return await self._embed_texts_ollama(texts)

        raise NotImplementedError(
            f"Embedding provider '{self.provider}' is not implemented in V1.1."
        )

    def _ollama_parse_vector(self, data: dict) -> List[float] | None:
        """New API: ``embeddings``: list[list[float]]; legacy: single ``embedding`` list."""
        raw: List[float] | None = None
        embs = data.get("embeddings")
        if isinstance(embs, list) and embs:
            first = embs[0]
            if isinstance(first, list) and first:
                raw = [float(x) for x in first]
        if raw is None:
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                raw = [float(x) for x in emb]
        return raw

    def _normalize_ollama_vec(self, vec: List[float]) -> List[float]:
        if len(vec) != self.dim:
            raise ValueError(
                f"Ollama embedding dim {len(vec)} != EMBEDDING_DIM {self.dim}. "
                f"Set EMBEDDING_DIM to match model (e.g. bge-m3 is often 1024) "
                f"and ensure pgvector column uses the same dimension."
            )
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def _embed_one_ollama(self, client: httpx.AsyncClient, text: str) -> List[float]:
        """Prefer ``POST /api/embed`` (``input``); legacy ``/api/embeddings`` last.

        502 on ``/api/embeddings`` is usually an Ollama/model crash, not a wrong host/port.
        """
        base = self._ollama_base
        t = text or ""
        last_err: str | None = None

        embed_opts: dict = {}
        if self._ollama_truncate:
            embed_opts["truncate"] = True

        for payload in (
            {"model": self.model_name, "input": t, **embed_opts},
            {"model": self.model_name, "input": [t], **embed_opts},
        ):
            r = await client.post(f"{base}/api/embed", json=payload)
            if r.status_code == 200:
                vec = self._ollama_parse_vector(r.json())
                if vec:
                    return self._normalize_ollama_vec(vec)
                last_err = f"/api/embed: empty embeddings, keys={list(r.json().keys())}"
            else:
                last_err = f"/api/embed: HTTP {r.status_code} {r.text[:300]}"

        leg: dict = {"model": self.model_name, "prompt": t}
        if self._ollama_truncate:
            leg["truncate"] = True
        r2 = await client.post(f"{base}/api/embeddings", json=leg)
        if r2.status_code >= 400:
            hint = (
                "（HTTP {0}）若 curl 正常而 Python 502，常见原因是系统代理（HTTP_PROXY）被 httpx 使用，"
                "本库默认 trust_env=False；若仍 502 请设 NO_PROXY=127.0.0.1,localhost 或检查 Ollama。"
                "也可重启 Ollama、确认 ollama list 含 {2}。试: curl -s {1}/api/embed -d \"...\""
            ).format(r2.status_code, base, self.model_name)
            raise RuntimeError(
                f"Ollama 嵌入失败。先请求: {last_err}；"
                f"回退 POST /api/embeddings 失败: {r2.status_code} {r2.text[:300]}。{hint}"
            ) from None
        vec2 = self._ollama_parse_vector(r2.json())
        if not vec2:
            raise RuntimeError(
                f"Ollama returned no embedding vector. {last_err}; "
                f"/api/embeddings keys={list(r2.json().keys())}"
            )
        return self._normalize_ollama_vec(vec2)

    async def _embed_texts_ollama(self, texts: List[str]) -> List[List[float]]:
        """Ollama embedding HTTP API.

        新版: ``POST {base}/api/embed``，字段 ``input``，返回 ``embeddings``。
        旧版: ``POST {base}/api/embeddings``，字段 ``prompt``，返回 ``embedding``。
        文档: https://docs.ollama.com/api/embed
        """
        out: List[List[float]] = []
        async with httpx.AsyncClient(
            timeout=self._ollama_timeout,
            trust_env=self._ollama_trust_env,
        ) as client:
            for text in texts:
                out.append(await self._embed_one_ollama(client, text or ""))
        return out

    def _ensure_bge_model_loaded(self) -> None:
        if self._model is not None:
            return
        # SentenceTransformers provides a convenient local runner for BGE embeddings.
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self._model = SentenceTransformer(self.model_name)
        # Keep dimension consistent with model output.
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            # If dimension probing fails, keep configured EMBEDDING_DIM.
            pass

    def _embed_texts_bge(self, texts: List[str]) -> List[List[float]]:
        self._ensure_bge_model_loaded()
        # normalize_embeddings=True makes cosine similarity stable.
        vectors = self._model.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.astype(float).tolist() for v in vectors]

    # ---- internal helpers -------------------------------------------------

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic bag-of-words hash embedding (no external deps).

        这不是语义模型，只是为了在没有外部服务时打通 end-to-end 流程。
        """

        dim = self.dim
        vec = [0.0] * dim
        if not text:
            return vec

        for raw_token in text.split():
            token = raw_token.strip()
            if not token:
                continue
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# Global singleton used by services
embedding_client = EmbeddingClient()

