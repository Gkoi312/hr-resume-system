"""Unit tests for semantic_chunk_matching — cosine band mapping, axis scoring, fusion."""

import math
import os
from typing import Any, Dict, List, Tuple

import pytest

from app.rag.vector_store import _cosine
from app.services.semantic_chunk_matching import (
    _axis_delivery_pure_cosine,
    _build_evidence_snippets,
    _cand_experience_chunks,
    _delivery_coverage_weights,
    _delivery_hybrid_enabled,
    _job_delivery_chunks,
    _rrf_k,
    _semantic_fusion_weights,
    _vec_nonempty,
    axis_delivery_score,
    axis_role_score,
    axis_skill_score,
    compute_semantic_scores_for_candidate_chunks,
    cosine_to_band_score,
    cosine_to_experience_score,
)


# ===================================================================
# helpers — synthetic vectors giving known cosine similarities
# ===================================================================

# Identical vectors → cosine = 1.0
V1 = [1.0, 2.0, 3.0]
V1_COPY = [1.0, 2.0, 3.0]

# Orthogonal → cosine = 0.0
V_ORTH = [1.0, 0.0, 0.0]
V_ORTH2 = [0.0, 1.0, 0.0]

# V1 vs V3 = [3,2,1]: dot=1*3+2*2+3*1=10, |V1|=sqrt(14)≈3.742, |V3|=sqrt(14)≈3.742, cos=10/14≈0.714
V3 = [3.0, 2.0, 1.0]
COS_V1_V3 = 10.0 / 14.0  # ≈ 0.714

# V_MID = [2,0,1]: dot with V1 = 1*2+2*0+3*1=5, |V_MID|=sqrt(5)≈2.236, cos=5/(3.742*2.236)≈0.598
V_MID = [2.0, 0.0, 1.0]
COS_V1_VMID = 5.0 / (math.sqrt(14) * math.sqrt(5))  # ≈ 0.598

# 0.85 cos pair: use [1,0,0.5] vs [1,0.2,0.3] — too hard to hand-calc precisely.
# Use two near-identical vectors with small perturbation for ~0.85.
V_HIGH = [1.0, 2.0, 3.0, 4.0]
V_HIGH_SIMILAR = [1.0, 2.0, 3.0, 3.5]  # slightly different → high cosine
COS_HIGH = _cosine(V_HIGH, V_HIGH_SIMILAR)  # compute once, likely 0.95+

# Empty
V_EMPTY: List[float] = []


def _chunk(profile_type: str, vec: List[float], semantic_text: str = "") -> Tuple:
    """Build a single chunk entry matching the (vec, meta) tuple convention."""
    return (profile_type, (vec, {"semantic_text": semantic_text}))


def _chunks(*entries: Tuple) -> Dict[str, Tuple[List[float], Dict[str, Any]]]:
    """Build a chunks dict from (profile_type, vec, semantic_text) tuples."""
    out: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
    for profile_type, vec, txt in entries:
        out[profile_type] = (list(vec), {"semantic_text": txt})
    return out


# ===================================================================
# _vec_nonempty
# ===================================================================
class TestVecNonempty:
    def test_nonempty(self):
        assert _vec_nonempty([1.0]) is True
        assert _vec_nonempty([0.0, 0.0]) is True

    def test_empty(self):
        assert _vec_nonempty([]) is False


# ===================================================================
# cosine_to_band_score
# ===================================================================
class TestCosineToBandScore:
    def test_zero_or_negative(self):
        assert cosine_to_band_score(0.0) == 0.0
        assert cosine_to_band_score(-0.5) == 0.0

    def test_boundaries(self):
        """Test each band boundary exactly."""
        # >= 0.84 → 95
        assert cosine_to_band_score(0.84) == 95.0
        assert cosine_to_band_score(0.90) == 95.0
        assert cosine_to_band_score(1.0) == 95.0
        # >= 0.78 → 85
        assert cosine_to_band_score(0.78) == 85.0
        assert cosine_to_band_score(0.80) == 85.0
        # >= 0.72 → 75
        assert cosine_to_band_score(0.72) == 75.0
        assert cosine_to_band_score(0.74) == 75.0
        # >= 0.66 → 65
        assert cosine_to_band_score(0.66) == 65.0
        assert cosine_to_band_score(0.70) == 65.0
        # < 0.66 → 55
        assert cosine_to_band_score(0.65) == 55.0
        assert cosine_to_band_score(0.40) == 55.0
        assert cosine_to_band_score(0.01) == 55.0

    def test_bands_are_discrete(self):
        """Band scores step through predefined levels only."""
        for sim in [0.85, 0.79, 0.73, 0.67, 0.50]:
            score = cosine_to_band_score(sim)
            assert score in (95.0, 85.0, 75.0, 65.0, 55.0, 0.0)


# ===================================================================
# cosine_to_experience_score
# ===================================================================
class TestCosineToExperienceScore:
    def test_zero_or_negative(self):
        assert cosine_to_experience_score(0.0) == 0.0
        assert cosine_to_experience_score(-0.1) == 0.0

    def test_high_bands_match_skill_axis(self):
        """High cosine values use same top band as cosine_to_band_score."""
        assert cosine_to_experience_score(0.84) == 95.0
        assert cosine_to_experience_score(1.0) == 95.0

    def test_upper_bands(self):
        assert cosine_to_experience_score(0.78) == 88.0
        assert cosine_to_experience_score(0.72) == 80.0
        assert cosine_to_experience_score(0.66) == 72.0

    def test_linear_interpolation_region(self):
        """Between 0.38 and 0.66, score is linearly interpolated."""
        # lo=0.38→38, hi=0.66→72
        assert cosine_to_experience_score(0.38) == 38.0  # lo boundary
        assert cosine_to_experience_score(0.66) == 72.0  # hi boundary (exact)
        # Midpoint: (0.38+0.66)/2 = 0.52 → (38+72)/2 = 55
        mid = (0.38 + 0.66) / 2
        mid_score = cosine_to_experience_score(mid)
        assert 54.0 <= mid_score <= 56.0  # Allow ~rounding

    def test_below_interpolation_range(self):
        """At or below 0.38, score floors at 38.0 (the linear range minimum)."""
        # 0.38 is the lo boundary → 38.0
        assert cosine_to_experience_score(0.38) == 38.0
        # Below 0.38 also floors at 38.0 (clamped to lo end of linear range)
        assert cosine_to_experience_score(0.30) == 38.0
        assert cosine_to_experience_score(0.01) == 38.0

    def test_monotonic(self):
        """Higher cosine should never give lower score."""
        values = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95]
        scores = [cosine_to_experience_score(v) for v in values]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Non-monotonic at {values[i]}→{values[i+1]}: {scores[i]}→{scores[i+1]}"
            )


# ===================================================================
# _job_delivery_chunks
# ===================================================================
class TestJobDeliveryChunks:
    def test_extracts_resp_chunks(self):
        chunks = _chunks(
            ("resp", V1, "职责1"),
            ("resp_1", V1, "职责2"),
            ("skill", V1, "技能"),
        )
        items = _job_delivery_chunks(chunks)
        pts = [it[0] for it in items]
        assert "resp" in pts
        assert "resp_1" in pts
        assert "skill" not in pts
        assert len(items) == 2

    def test_falls_back_to_jd_raw(self):
        """When no resp chunks found, use jd_raw."""
        chunks = _chunks(
            ("jd_raw", V1, "完整JD"),
            ("skill", V1, "技能"),
        )
        items = _job_delivery_chunks(chunks)
        assert len(items) == 1
        assert items[0][0] == "jd_raw"

    def test_empty_when_nothing(self):
        chunks = _chunks(("skill", V1, "技能"))
        items = _job_delivery_chunks(chunks)
        assert items == []

    def test_skips_empty_vectors(self):
        chunks = _chunks(
            ("resp", V_EMPTY, "空职责"),
            ("resp_1", V1, "有效职责"),
        )
        items = _job_delivery_chunks(chunks)
        assert len(items) == 1
        assert items[0][0] == "resp_1"

    def test_jd_raw_fallback_skips_empty(self):
        chunks = _chunks(("jd_raw", V_EMPTY, "空JD"))
        items = _job_delivery_chunks(chunks)
        assert items == []


# ===================================================================
# _cand_experience_chunks
# ===================================================================
class TestCandExperienceChunks:
    def test_extracts_proj_and_work(self):
        chunks = _chunks(
            ("proj_0", V1, "项目A"),
            ("work_0", V1, "实习A"),
            ("work_1", V1, "实习B"),
            ("skill", V1, "技能"),
            ("cand_role", V1, "角色"),
        )
        items = _cand_experience_chunks(chunks)
        pts = [it[0] for it in items]
        assert set(pts) == {"proj_0", "work_0", "work_1"}
        assert len(items) == 3

    def test_skips_empty_vectors(self):
        chunks = _chunks(
            ("proj_0", V_EMPTY, "空项目"),
            ("work_0", V1, "有效实习"),
        )
        items = _cand_experience_chunks(chunks)
        assert len(items) == 1
        assert items[0][0] == "work_0"

    def test_empty_when_no_experience(self):
        chunks = _chunks(("skill", V1, "技能"))
        items = _cand_experience_chunks(chunks)
        assert items == []


# ===================================================================
# axis_skill_score
# ===================================================================
class TestAxisSkillScore:
    def test_perfect_match(self):
        job = _chunks(("skill", V1, "Go Python"))
        cand = _chunks(("skill", V1_COPY, "Go Python"))
        band, sim, ok = axis_skill_score(job, cand)
        assert ok is True
        assert sim == pytest.approx(1.0)
        assert band == 95.0

    def test_no_job_skill_chunk(self):
        job = _chunks(("role", V1, "后端"))
        cand = _chunks(("skill", V1, "Go"))
        band, sim, ok = axis_skill_score(job, cand)
        assert ok is False
        assert band == 0.0
        assert sim is None

    def test_no_cand_skill_chunk(self):
        job = _chunks(("skill", V1, "Go"))
        cand = _chunks(("cand_role", V1, "后端"))
        band, sim, ok = axis_skill_score(job, cand)
        assert ok is False
        assert band == 0.0
        assert sim is None

    def test_empty_vectors(self):
        job = _chunks(("skill", V_EMPTY, ""))
        cand = _chunks(("skill", V1, "Go"))
        band, sim, ok = axis_skill_score(job, cand)
        assert ok is False
        assert band == 0.0

    def test_partial_similarity(self):
        job = _chunks(("skill", V1, ""))
        cand = _chunks(("skill", V3, ""))
        band, sim, ok = axis_skill_score(job, cand)
        assert ok is True
        assert sim == pytest.approx(COS_V1_V3, rel=1e-3)
        # COS_V1_V3 ≈ 0.714 → band: >= 0.66 → 65
        assert band == 65.0


# ===================================================================
# axis_role_score
# ===================================================================
class TestAxisRoleScore:
    def test_match(self):
        job = _chunks(("role", V1, "后端开发"))
        cand = _chunks(("cand_role", V1_COPY, "后端开发"))
        band, sim, ok = axis_role_score(job, cand)
        assert ok is True
        assert sim == pytest.approx(1.0)
        assert band == 95.0

    def test_no_cand_role(self):
        job = _chunks(("role", V1, "后端"))
        cand = _chunks(("skill", V1, ""))
        band, sim, ok = axis_role_score(job, cand)
        assert ok is False
        assert band == 0.0


# ===================================================================
# _axis_delivery_pure_cosine
# ===================================================================
class TestAxisDeliveryPureCosine:
    @staticmethod
    def _make_items(prefix: str, count: int, vectors: List[List[float]]) -> List[Tuple]:
        return [
            (f"{prefix}_{i}", vec, {"semantic_text": f"{prefix} text {i}"})
            for i, vec in enumerate(vectors)
        ]

    def test_single_job_single_cand_perfect(self):
        job_items = self._make_items("resp", 1, [V1])
        cand_items = self._make_items("proj", 1, [V1_COPY])
        cos_matrix = [[_cosine(jv, cv) for cv in [V1_COPY]] for jv in [V1]]
        detail: Dict[str, Any] = {}
        score, detail = _axis_delivery_pure_cosine(job_items, cand_items, cos_matrix, detail)
        assert detail["axis_empty"] is False
        assert detail["delivery_fusion"] == "cosine_only"
        # Perfect match → cosine 1.0 → experience_score(1.0) = 95
        assert score == 95.0

    def test_single_job_single_cand_mismatch(self):
        job_items = self._make_items("resp", 1, [V1])
        cand_items = self._make_items("proj", 1, [V_ORTH2])
        cos_matrix = [[_cosine(V1, V_ORTH2)]]  # ≈ 0.534
        detail: Dict[str, Any] = {}
        score, _ = _axis_delivery_pure_cosine(job_items, cand_items, cos_matrix, detail)
        # cos(0.534) → experience_score via linear interp → between 38-72 range
        assert 0 < score < 72

    def test_multiple_job_multiple_cand(self):
        job_items = self._make_items("resp", 2, [V1, V_ORTH])
        cand_items = self._make_items("proj", 2, [V1_COPY, V_ORTH2])
        # cos matrix: [1.0, 0.534], [0.267, 1.0]
        cos_matrix = [
            [_cosine(jv, cv) for cv in [V1_COPY, V_ORTH2]]
            for jv in [V1, V_ORTH]
        ]
        detail: Dict[str, Any] = {}
        score, detail = _axis_delivery_pure_cosine(job_items, cand_items, cos_matrix, detail)
        assert score >= 0.0
        assert detail["axis_empty"] is False
        assert detail["delivery_topk"] > 0

    def test_high_cosine_gives_high_score(self):
        """V_HIGH vs V_HIGH_SIMILAR should give high cosine → high experience score."""
        job_items = self._make_items("resp", 1, [V_HIGH])
        cand_items = self._make_items("proj", 1, [V_HIGH_SIMILAR])
        cos = _cosine(V_HIGH, V_HIGH_SIMILAR)
        cos_matrix = [[cos]]
        detail: Dict[str, Any] = {}
        score, _ = _axis_delivery_pure_cosine(job_items, cand_items, cos_matrix, detail)
        expected = cosine_to_experience_score(cos)
        # Single job × single cand → job_mean = cand_mean = expected → blended same
        assert score == pytest.approx(expected, rel=0.01)


# ===================================================================
# axis_delivery_score (hybrid mode)
# ===================================================================
class TestAxisDeliveryScore:
    def test_empty_on_one_side(self):
        """If job has no delivery chunks, returns 0."""
        job = _chunks(("skill", V1, ""))
        cand = _chunks(("proj_0", V1, "项目"))
        score, detail = axis_delivery_score(job, cand)
        assert score == 0.0
        assert detail["axis_empty"] is True

    def test_empty_on_cand_side(self):
        job = _chunks(("resp", V1, "职责"))
        cand = _chunks(("skill", V1, ""))
        score, detail = axis_delivery_score(job, cand)
        assert score == 0.0
        assert detail["axis_empty"] is True

    def test_single_pair_hybrid(self):
        """One resp ↔ one proj: hybrid still works (BM25 degenerates but RRF handles it)."""
        job = _chunks(("resp", V1, "负责后端系统开发与维护"))
        cand = _chunks(("proj_0", V1_COPY, "负责后端系统开发与维护"))
        score, detail = axis_delivery_score(job, cand)
        assert detail["axis_empty"] is False
        assert score > 0
        assert "delivery_alignments" in detail

    def test_jd_raw_fallback(self):
        """When no resp chunks, jd_raw is used."""
        job = _chunks(("jd_raw", V1, "开发工程师，负责平台开发"))
        cand = _chunks(("work_0", V1_COPY, "开发工程师，负责平台开发"))
        score, detail = axis_delivery_score(job, cand)
        assert detail["axis_empty"] is False
        assert score > 0

    def test_multiple_chunks_uses_rrf_fusion(self):
        job = _chunks(
            ("resp", V1, "构建数据pipeline"),
            ("resp_1", V_ORTH, "前端页面开发"),
        )
        cand = _chunks(
            ("proj_0", V1_COPY, "构建数据pipeline项目"),
            ("work_0", V_ORTH2, "前端开发实习"),
        )
        score, detail = axis_delivery_score(job, cand)
        assert detail["axis_empty"] is False
        assert detail.get("delivery_fusion") == "rrf"
        assert score > 0


# ===================================================================
# compute_semantic_scores_for_candidate_chunks (fusion)
# ===================================================================
class TestComputeSemanticScoresForCandidateChunks:
    def test_full_chunks(self):
        """All three axes present → full fusion."""
        job = _chunks(
            ("skill", V_HIGH, "Go Python"),
            ("role", V_HIGH, "后端开发"),
            ("resp", V_HIGH, "构建微服务"),
        )
        cand = _chunks(
            ("skill", V_HIGH_SIMILAR, "Go Python"),
            ("cand_role", V_HIGH_SIMILAR, "后端开发"),
            ("proj_0", V_HIGH_SIMILAR, "微服务项目"),
        )
        score, detail = compute_semantic_scores_for_candidate_chunks(job, cand)
        assert detail["semantic_status"] == "available"
        assert detail["skill_axis_complete"] is True
        assert detail["role_axis_complete"] is True
        assert detail["delivery_axis_complete"] is True
        # Fusion weights: default skill=0.2, delivery=0.8, role=0.0
        assert detail["semantic_weight_skill"] == pytest.approx(0.2, rel=0.1)
        assert detail["semantic_weight_delivery"] == pytest.approx(0.8, rel=0.1)
        assert detail["semantic_weight_role"] == pytest.approx(0.0, abs=0.01)
        assert 0.0 <= score <= 100.0

    def test_not_indexed_candidate(self):
        """Empty cand chunks → not_indexed."""
        job = _chunks(("skill", V1, ""))
        cand: Dict[str, Tuple] = {}
        score, detail = compute_semantic_scores_for_candidate_chunks(job, cand)
        assert detail["semantic_status"] == "not_indexed"
        # All axes incomplete → score 0
        assert score == 0.0

    def test_only_skill_axis(self):
        """When only skill chunks exist on both sides — delivery/role missing get 0.
        Default weights: skill=0.2, delivery=0.8, so only skill axis → 0.2 * 95 = 19."""
        job = _chunks(("skill", V1, "Go"))
        cand = _chunks(("skill", V1_COPY, "Go"))
        score, detail = compute_semantic_scores_for_candidate_chunks(job, cand)
        assert detail["skill_axis_complete"] is True
        assert detail["delivery_axis_complete"] is False
        # With default weights (skill 0.2), only 20% of the skill score contributes
        assert score == 19.0

    def test_evidence_snippets_present(self):
        job = _chunks(
            ("skill", V1, "Go Python Docker"),
            ("resp", V1, "构建后端服务"),
        )
        cand = _chunks(
            ("skill", V1_COPY, "Go Python Docker"),
            ("proj_0", V1_COPY, "构建后端服务项目"),
        )
        _, detail = compute_semantic_scores_for_candidate_chunks(job, cand)
        snippets = detail.get("evidence_snippets", [])
        assert len(snippets) > 0
        # Should include skill snippet
        src_types = [s["source_type"] for s in snippets]
        assert "skill" in src_types


# ===================================================================
# _build_evidence_snippets
# ===================================================================
class TestBuildEvidenceSnippets:
    def test_skill_snippet(self):
        job = _chunks(("skill", V1, ""))
        cand = _chunks(("skill", V1_COPY, "Go Python"))
        snippets = _build_evidence_snippets(job, cand, 0.85, None, {})
        assert len(snippets) >= 1
        skill_snip = [s for s in snippets if s["source_type"] == "skill"]
        assert len(skill_snip) == 1
        assert skill_snip[0]["similarity"] == 0.85

    def test_role_snippet(self):
        job = _chunks(("role", V1, ""))
        cand = _chunks(("cand_role", V1, "后端工程师"))
        snippets = _build_evidence_snippets(job, cand, None, 0.90, {})
        role_snip = [s for s in snippets if s["source_type"] == "role"]
        assert len(role_snip) == 1
        assert role_snip[0]["similarity"] == 0.90

    def test_no_delivery_snippet_when_axis_empty(self):
        job = _chunks(("skill", V1, ""))
        cand = _chunks(("skill", V1, "Go"))
        snippets = _build_evidence_snippets(job, cand, 0.8, None, {"axis_empty": True})
        src_types = [s["source_type"] for s in snippets]
        assert "delivery" not in src_types

    def test_delivery_snippet_from_alignments(self):
        snippets = _build_evidence_snippets(
            _chunks(("resp", V1, "JD职责")),
            _chunks(("proj_0", V1, "经历")),
            None,
            None,
            {
                "axis_empty": False,
                "delivery_alignments": [
                    {
                        "job_profile_type": "resp",
                        "cand_profile_type": "proj_0",
                        "cosine": 0.75,
                        "rrf": 0.85,
                        "bm25_degenerate": False,
                        "job_text_snippet": "JD职责",
                        "cand_text_snippet": "项目经历描述",
                    }
                ],
            },
        )
        assert len(snippets) == 1
        assert snippets[0]["source_type"] == "delivery"
        assert snippets[0]["similarity"] == 0.75

    def test_empty_when_no_data(self):
        snippets = _build_evidence_snippets({}, {}, None, None, {})
        assert snippets == []


# ===================================================================
# _delivery_hybrid_enabled — env toggle
# ===================================================================
class TestDeliveryHybridEnabled:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_DELIVERY_HYBRID", raising=False)
        assert _delivery_hybrid_enabled() is True

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_DELIVERY_HYBRID", "0")
        assert _delivery_hybrid_enabled() is False
        monkeypatch.setenv("SEMANTIC_DELIVERY_HYBRID", "false")
        assert _delivery_hybrid_enabled() is False
        monkeypatch.setenv("SEMANTIC_DELIVERY_HYBRID", "off")
        assert _delivery_hybrid_enabled() is False


# ===================================================================
# _rrf_k
# ===================================================================
class TestRRFK:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_RRF_K", raising=False)
        assert _rrf_k() == 60

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_RRF_K", "120")
        assert _rrf_k() == 120

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_RRF_K", "abc")
        assert _rrf_k() == 60

    def test_below_1_clamped(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_RRF_K", "0")
        assert _rrf_k() == 1


# ===================================================================
# _semantic_fusion_weights
# ===================================================================
class TestSemanticFusionWeights:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_WEIGHT_SKILL", raising=False)
        monkeypatch.delenv("SEMANTIC_WEIGHT_DELIVERY", raising=False)
        monkeypatch.delenv("SEMANTIC_WEIGHT_ROLE", raising=False)
        w_s, w_d, w_r = _semantic_fusion_weights()
        assert w_s + w_d + w_r == pytest.approx(1.0)
        assert w_s == pytest.approx(0.2)
        assert w_d == pytest.approx(0.8)
        assert w_r == pytest.approx(0.0)

    def test_custom_weights_normalized(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_WEIGHT_SKILL", "0.5")
        monkeypatch.setenv("SEMANTIC_WEIGHT_DELIVERY", "0.5")
        monkeypatch.setenv("SEMANTIC_WEIGHT_ROLE", "0.0")
        w_s, w_d, w_r = _semantic_fusion_weights()
        assert w_s + w_d + w_r == pytest.approx(1.0)
        assert w_s == pytest.approx(0.5)
        assert w_d == pytest.approx(0.5)

    def test_all_zero_falls_back(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_WEIGHT_SKILL", "0")
        monkeypatch.setenv("SEMANTIC_WEIGHT_DELIVERY", "0")
        monkeypatch.setenv("SEMANTIC_WEIGHT_ROLE", "0")
        w_s, w_d, w_r = _semantic_fusion_weights()
        # Falls back to defaults (0.20, 0.80, 0.00)
        assert w_s == pytest.approx(0.2)
        assert w_d == pytest.approx(0.8)
        assert w_r == pytest.approx(0.0)


# ===================================================================
# _delivery_coverage_weights
# ===================================================================
class TestDeliveryCoverageWeights:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_DELIVERY_JOB_COVERAGE", raising=False)
        wj, wc = _delivery_coverage_weights()
        assert wj == pytest.approx(0.62)
        assert wc == pytest.approx(0.38)

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_DELIVERY_JOB_COVERAGE", "0.7")
        wj, wc = _delivery_coverage_weights()
        assert wj == pytest.approx(0.7)
        assert wc == pytest.approx(0.3)

    def test_out_of_range_clamped(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_DELIVERY_JOB_COVERAGE", "2.0")
        wj, wc = _delivery_coverage_weights()
        assert wj == 1.0
        assert wc == 0.0

    def test_negative_clamped(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_DELIVERY_JOB_COVERAGE", "-1")
        wj, wc = _delivery_coverage_weights()
        assert wj == 0.0
        assert wc == 1.0
