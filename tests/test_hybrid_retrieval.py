"""Unit tests for BM25 + RRF helpers."""

import pytest

from app.rag.hybrid_retrieval import (
    BM25Okapi,
    competition_ranks,
    compute_rrf_matrix,
    tokenize,
)


def test_tokenize_mixed():
    t = tokenize("Python 后端 + Rust 开发")
    assert "python" in t
    assert "rust" in t
    assert "开发" in t
    assert "后端" in t


def test_tokenize_english_words_not_per_char():
    t = tokenize("hello world API")
    assert t == ["hello", "world", "api"]


def test_tokenize_numbers_and_version():
    t = tokenize("v1.2.3 2024 3.14")
    assert "v1.2.3" in t
    assert "2024" in t
    assert "3.14" in t


def test_competition_ranks_ties():
    r = competition_ranks([10.0, 10.0, 5.0], higher_is_better=True)
    assert r[0] == r[1] == 1
    assert r[2] == 3


def test_bm25_prefers_lexical_match():
    corpus = [
        tokenize("销售与客户关系"),
        tokenize("python fastapi 后端开发"),
    ]
    q = tokenize("python 后端")
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q)
    assert scores[1] > scores[0]


def test_compute_rrf_matrix_smoke():
    cos_matrix = [[0.9, 0.2], [0.3, 0.85]]
    job_tok = [tokenize("python api"), tokenize("react ui")]
    doc_tok = [tokenize("django python"), tokenize("react typescript")]
    rrf, deg, bm25_rows = compute_rrf_matrix(
        cos_matrix, job_tok, doc_tok, rrf_k=60
    )
    assert len(rrf) == 2 and len(rrf[0]) == 2
    assert len(deg) == 2
    assert len(bm25_rows) == 2


def test_compute_rrf_matrix_empty_matrix_requires_empty_inputs():
    rrf, deg, bm25 = compute_rrf_matrix([], [], [], rrf_k=60)
    assert rrf == [] and deg == [] and bm25 == []


def test_compute_rrf_matrix_rejects_mismatched_dimensions():
    cos = [[0.9, 0.2], [0.3, 0.85]]
    job_ok = [tokenize("a"), tokenize("b")]
    doc_ok = [tokenize("x"), tokenize("y")]
    with pytest.raises(ValueError, match="job_queries_tokenized length"):
        compute_rrf_matrix(cos, [job_ok[0]], doc_ok, rrf_k=60)
    with pytest.raises(ValueError, match="cand_docs_tokenized length"):
        compute_rrf_matrix(cos, job_ok, [doc_ok[0]], rrf_k=60)
    with pytest.raises(ValueError, match="rectangular"):
        compute_rrf_matrix([[0.1, 0.2], [0.3]], job_ok, doc_ok, rrf_k=60)


def test_compute_rrf_matrix_empty_matrix_rejects_nonempty_queries():
    with pytest.raises(ValueError, match="cos_matrix is empty"):
        compute_rrf_matrix([], [[]], [], rrf_k=60)
    with pytest.raises(ValueError, match="cos_matrix is empty"):
        compute_rrf_matrix([], [], [[]], rrf_k=60)
