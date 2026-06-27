"""Tests for delivery alignment enrichment on match explanations."""

from app.services.matching_service import _enrich_delivery_alignments


def test_enrich_alignments_shared_terms():
    raw = [
        {
            "job_profile_type": "resp_0",
            "cand_profile_type": "work_0",
            "job_text_snippet": "python 后端开发",
            "cand_text_snippet": "python fastapi 后端",
            "cosine": 0.8,
            "bm25": 1.2,
            "rank_cos": 1,
            "rank_bm25": 1,
            "rrf": 0.032786,
            "bm25_degenerate": False,
        }
    ]
    items = _enrich_delivery_alignments(raw)
    assert len(items) == 1
    it = items[0]
    assert it.job_profile_type == "resp_0"
    assert it.cand_profile_type == "work_0"
    assert "python" in it.shared_terms
    assert "后端" in it.shared_terms


def test_enrich_skips_non_dict_rows():
    assert _enrich_delivery_alignments([None, "x"]) == []
