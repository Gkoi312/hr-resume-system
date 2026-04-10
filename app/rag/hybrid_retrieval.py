"""Lexical + dense hybrid helpers for chunk-level retrieval (Delivery axis).

BM25 over candidate experience texts; fusion via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import jieba

_DEFAULT_USER_WORD_FREQ = 100_000

_USERWORDS_LOADED = False


def _load_jieba_user_words() -> None:
    """从同目录 ``jieba_userdict.txt`` 注入自定义词，进程内只执行一次。"""
    global _USERWORDS_LOADED
    if _USERWORDS_LOADED:
        return
    path = Path(__file__).resolve().with_name("jieba_userdict.txt")
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            jieba.add_word(line, freq=_DEFAULT_USER_WORD_FREQ)
    _USERWORDS_LOADED = True


_load_jieba_user_words()

# 仅标点/空白，不参与 BM25
_PUNCT_OR_SPACE_ONLY = re.compile(r"^[^\w\u4e00-\u9fff]+$")
# 英文/数字技术词（拉丁连续段内从左向右匹配，不依赖 jieba 对英文的切分）
_ASCII_TECH = re.compile(r"[a-zA-Z][a-zA-Z0-9.+#_-]*|[0-9][a-zA-Z0-9.+#_-]*")


def _is_cjk(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def _emit_jieba_cjk_chunk(chunk: str, out: List[str]) -> None:
    for w in jieba.lcut(chunk):
        w = w.strip()
        if not w or _PUNCT_OR_SPACE_ONLY.match(w):
            continue
        if _ASCII_TECH.fullmatch(w):
            out.append(w.lower())
            continue
        if re.search(r"[\u4e00-\u9fff]", w):
            out.append(w)
            continue
        for m in _ASCII_TECH.finditer(w.lower()):
            out.append(m.group(0))


def tokenize(text: str) -> List[str]:
    """中英混合：连续汉字块用 jieba；英文/数字在拉丁连续段内用正则稳定切词。

    拉丁段与整段 jieba 不同，可避免英文被 jieba 切成单字母；纯数字、小数、版本号
    等与 ``_ASCII_TECH`` 一致。用于 BM25：重复词保留（词频有意义）。
    """
    if not text or not str(text).strip():
        return []
    s = str(text).strip()
    out: List[str] = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if _is_cjk(ch):
            j = i + 1
            while j < n and _is_cjk(s[j]):
                j += 1
            _emit_jieba_cjk_chunk(s[i:j], out)
            i = j
            continue
        m = _ASCII_TECH.match(s, i)
        if m:
            out.append(m.group(0).lower())
            i = m.end()
            continue
        i += 1
    return out


class BM25Okapi:
    """Okapi BM25 over a fixed tokenized corpus; query is token list per call."""

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus_size = len(corpus)
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (
            sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        )
        self.doc_freqs: List[dict[str, int]] = []
        nd: dict[str, int] = {}
        for doc in corpus:
            f: dict[str, int] = {}
            for w in doc:
                f[w] = f.get(w, 0) + 1
            self.doc_freqs.append(f)
            for w in f:
                nd[w] = nd.get(w, 0) + 1
        self.idf: dict[str, float] = {}
        for word, freq in nd.items():
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query: Sequence[str]) -> List[float]:
        if not self.corpus_size:
            return []
        scores = [0.0] * self.corpus_size
        avgdl = self.avgdl or 1.0
        for idx, doc in enumerate(self.doc_freqs):
            dl = self.doc_len[idx]
            s = 0.0
            for q in query:
                if q not in doc:
                    continue
                freq = doc[q]
                idf = self.idf.get(q, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                s += idf * (freq * (self.k1 + 1.0) / denom)
            scores[idx] = s
        return scores


def competition_ranks(scores: Sequence[float], *, higher_is_better: bool = True) -> List[int]:
    """1-based ranks; ties share rank; next rank skips (1,1,3)."""
    n = len(scores)
    if n == 0:
        return []
    idx = sorted(range(n), key=lambda i: scores[i], reverse=higher_is_better)
    ranks = [0] * n
    pos = 0
    current_rank = 1
    while pos < n:
        j = pos + 1
        while j < n and scores[idx[j]] == scores[idx[pos]]:
            j += 1
        for t in range(pos, j):
            ranks[idx[t]] = current_rank
        current_rank += j - pos
        pos = j
    return ranks


def bm25_scores_degenerate(scores: Sequence[float], *, eps: float = 1e-9) -> bool:
    if not scores:
        return True
    lo = min(scores)
    hi = max(scores)
    return (hi - lo) <= eps


def rrf_pair(rank_a: int, rank_b: int, k: int) -> float:
    return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)


def compute_rrf_matrix(
    cos_matrix: List[List[float]],
    job_queries_tokenized: List[List[str]],
    cand_docs_tokenized: List[List[str]],
    *,
    rrf_k: int,
) -> Tuple[List[List[float]], List[bool], List[List[float]]]:
    """
    cos_matrix[j][d]: cosine between job delivery j and cand experience d.

    Returns (rrf[j][d], bm25_degenerate_per_row[j], bm25_score_row[j][d]).
    """
    j_count = len(cos_matrix)
    if j_count == 0:
        if len(job_queries_tokenized) != 0:
            raise ValueError(
                "cos_matrix is empty but job_queries_tokenized is non-empty "
                f"({len(job_queries_tokenized)} rows)"
            )
        if len(cand_docs_tokenized) != 0:
            raise ValueError(
                "cos_matrix is empty but cand_docs_tokenized is non-empty "
                f"({len(cand_docs_tokenized)} docs)"
            )
        return [], [], []

    d_count = len(cos_matrix[0])
    for r, row in enumerate(cos_matrix):
        if len(row) != d_count:
            raise ValueError(
                "cos_matrix must be rectangular: "
                f"row {r} has length {len(row)}, expected {d_count}"
            )

    if d_count == 0:
        return [], [], []

    if len(job_queries_tokenized) != j_count:
        raise ValueError(
            "job_queries_tokenized length must equal cos_matrix row count: "
            f"got {len(job_queries_tokenized)}, expected {j_count}"
        )
    if len(cand_docs_tokenized) != d_count:
        raise ValueError(
            "cand_docs_tokenized length must equal cos_matrix column count: "
            f"got {len(cand_docs_tokenized)}, expected {d_count}"
        )

    bm25_engine = BM25Okapi(cand_docs_tokenized)
    rrf: List[List[float]] = []
    degenerate_flags: List[bool] = []
    bm25_rows: List[List[float]] = []

    for j in range(j_count):
        row_cos = cos_matrix[j]
        rank_cos = competition_ranks(row_cos, higher_is_better=True)
        q = job_queries_tokenized[j]
        if not q:
            bm25_row = [0.0] * d_count
        else:
            bm25_row = bm25_engine.get_scores(q)
        bm25_rows.append(bm25_row)
        deg = bm25_scores_degenerate(bm25_row)
        degenerate_flags.append(deg)
        rank_bm25 = competition_ranks(bm25_row, higher_is_better=True)
        row_rrf: List[float] = []
        for d in range(d_count):
            if deg:
                row_rrf.append(1.0 / (rrf_k + rank_cos[d]))
            else:
                row_rrf.append(rrf_pair(rank_cos[d], rank_bm25[d], rrf_k))
        rrf.append(row_rrf)

    return rrf, degenerate_flags, bm25_rows
