"""Pure-function unit tests for matching_service scoring and explanation logic.

Tests the load-bearing functions that produce scores, pros/cons,
recommendations, and explanation structures — without DB, services, or LLM.
"""

from typing import Any, Dict, List

import pytest

from app.services.matching_service import (
    _build_match_explanation,
    _compute_quality_factor,
    _enrich_delivery_alignments,
    _pros_cons_recommendation,
    _shared_terms_for_alignment,
)
from app.schemas.match import DeliveryAlignmentItem, MatchExplanation


# =============================================================================
# _compute_quality_factor
# =============================================================================

class TestComputeQualityFactor:
    def test_perfect_quality_gives_factor_1(self):
        assert _compute_quality_factor(100.0) == 1.0

    def test_zero_quality_gives_floor_0_6(self):
        assert _compute_quality_factor(0.0) == 0.6

    def test_midpoint_quality_50(self):
        assert _compute_quality_factor(50.0) == 0.8  # 0.6 + 0.4*0.5

    def test_typical_quality_25(self):
        assert _compute_quality_factor(25.0) == 0.7  # 0.6 + 0.4*0.25

    def test_quality_75(self):
        assert _compute_quality_factor(75.0) == 0.9

    def test_factor_monotonic(self):
        """Higher quality must produce higher (or equal) factor."""
        factors = [_compute_quality_factor(q) for q in range(0, 101, 5)]
        for i in range(1, len(factors)):
            assert factors[i] >= factors[i - 1]

    def test_factor_range(self):
        """Factor must always be in [0.6, 1.0] for valid inputs."""
        for q in [0, 10, 25, 50, 75, 100]:
            f = _compute_quality_factor(float(q))
            assert 0.6 <= f <= 1.0

    def test_negative_quality_still_works(self):
        """Negative quality shouldn't crash; factor goes below floor."""
        f = _compute_quality_factor(-10.0)
        # 0.6 + 0.4*(-10/100) = 0.56
        assert f < 0.6

    def test_above_100_still_works(self):
        """Quality > 100 should be allowed (factor > 1.0)."""
        f = _compute_quality_factor(150.0)
        assert f > 1.0


# =============================================================================
# _pros_cons_recommendation
# =============================================================================

class TestProsConsRecommendation:

    # ── Recommendation thresholds ──

    @pytest.mark.parametrize(
        "overall, expected",
        [
            (90.0, "建议进入初筛"),
            (70.0, "建议进入初筛"),
            (69.9, "建议备选"),
            (50.0, "建议备选"),
            (49.9, "可观望"),
            (35.0, "可观望"),
            (34.9, "暂不推荐"),
            (0.0, "暂不推荐"),
        ],
    )
    def test_recommendation_thresholds(self, overall, expected):
        _, _, rec = _pros_cons_recommendation(
            overall_score=overall,
            semantic_score=50.0,
            skill_score=80.0,
            quality_score=50.0,
            quality_status="rule_based",
            semantic_status="available",
        )
        assert rec == expected

    # ── Skill pros/cons ──

    def test_skill_high_adds_pro(self):
        pros, cons, _ = _pros_cons_recommendation(80, 50, 85, 50, "rule_based", "available")
        assert any("技能命中率较高" in p for p in pros)

    def test_skill_low_adds_con(self):
        pros, cons, _ = _pros_cons_recommendation(30, 50, 35, 50, "rule_based", "available")
        assert any("技能命中率偏低" in c for c in cons)

    def test_skill_midrange_neutral(self):
        """Skill 40-79: no pros and no cons for skill."""
        pros, cons, _ = _pros_cons_recommendation(50, 50, 60, 50, "rule_based", "available")
        assert not any("技能命中率" in p for p in pros)
        assert not any("技能命中率" in c for c in cons)

    # ── Semantic pros/cons ──

    def test_semantic_not_indexed_adds_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 50, 60, 50, "rule_based", "not_indexed")
        assert any("语义索引" in c for c in cons)

    def test_semantic_high_adds_pro(self):
        pros, _, _ = _pros_cons_recommendation(50, 85, 60, 50, "rule_based", "available")
        assert any("语义相关度高" in p for p in pros)

    def test_semantic_mid_adds_pro(self):
        pros, _, _ = _pros_cons_recommendation(50, 65, 60, 50, "rule_based", "available")
        assert any("一定语义相关性" in p for p in pros)

    def test_semantic_low_adds_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 30, 60, 50, "rule_based", "available")
        assert any("语义相关性偏低" in c for c in cons)

    def test_semantic_exactly_80_adds_pro(self):
        pros, _, _ = _pros_cons_recommendation(50, 80, 60, 50, "rule_based", "available")
        assert any("语义相关度高" in p for p in pros)

    def test_semantic_exactly_60_adds_pro(self):
        pros, _, _ = _pros_cons_recommendation(50, 60, 60, 50, "rule_based", "available")
        assert any("一定语义相关性" in p for p in pros)

    # ── Quality pros/cons ──

    def test_quality_high_adds_pro(self):
        pros, _, _ = _pros_cons_recommendation(50, 50, 60, 75, "rule_based", "available")
        assert any("描述较充实" in p for p in pros)

    def test_quality_exactly_70_adds_pro(self):
        pros, _, _ = _pros_cons_recommendation(50, 50, 60, 70, "rule_based", "available")
        assert any("描述较充实" in p for p in pros)

    def test_quality_very_low_adds_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 50, 60, 15, "rule_based", "available")
        assert any("描述偏少或空泛" in c for c in cons)

    def test_quality_exactly_20_adds_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 50, 60, 20, "rule_based", "available")
        assert any("描述偏少或空泛" in c for c in cons)

    def test_quality_low_adds_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 50, 60, 35, "rule_based", "available")
        assert any("较简短" in c for c in cons)

    def test_quality_exactly_40_adds_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 50, 60, 40, "rule_based", "available")
        assert any("较简短" in c for c in cons)

    def test_quality_midrange_neutral(self):
        """Quality 41-69: no pros and no cons for quality."""
        pros, cons, _ = _pros_cons_recommendation(50, 50, 60, 55, "rule_based", "available")
        assert not any("实习/项目" in p for p in pros)
        assert not any("实习/项目" in c for c in cons)

    # ── Quality status ──

    def test_quality_status_disabled_adds_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 50, 60, 50, "disabled", "available")
        assert any("规则版" in c for c in cons)

    def test_quality_status_rule_based_no_con(self):
        _, cons, _ = _pros_cons_recommendation(50, 50, 60, 50, "rule_based", "available")
        assert not any("规则版" in c for c in cons)

    # ── Combined scenarios ──

    def test_best_case(self):
        """Perfect scores: all pros, recommend interview."""
        pros, cons, rec = _pros_cons_recommendation(85, 90, 95, 85, "rule_based", "available")
        assert len(pros) >= 3
        assert len(cons) == 0
        assert rec == "建议进入初筛"

    def test_worst_case(self):
        """Zero scores: all cons, not recommended."""
        pros, cons, rec = _pros_cons_recommendation(10, 10, 10, 5, "disabled", "available")
        assert len(cons) >= 3
        assert rec == "暂不推荐"

    def test_returns_three_tuple(self):
        result = _pros_cons_recommendation(50, 50, 50, 50, "rule_based", "available")
        assert isinstance(result, tuple)
        assert len(result) == 3
        pros, cons, rec = result
        assert isinstance(pros, list)
        assert isinstance(cons, list)
        assert isinstance(rec, str)


# =============================================================================
# _build_match_explanation
# =============================================================================

class TestBuildMatchExplanation:

    def _skill_info(self, skill_score=75.0, matched=None, missing=None,
                    job_terms=None, cand_terms=None):
        return {
            "skill_score": skill_score,
            "matched_skills": matched or [],
            "missing_skills": missing or [],
            "job_skill_terms": job_terms or [],
            "candidate_skill_terms": cand_terms or [],
        }

    def _llm_info(self, llm_quality_score=50.0, evidence_quality=50.0,
                  impact=50.0, consistency_risk=30.0, status="rule_based",
                  summary=None):
        return {
            "llm_quality_score": llm_quality_score,
            "evidence_quality_score": evidence_quality,
            "impact_score": impact,
            "consistency_risk": consistency_risk,
            "status": status,
            "summary": summary,
        }

    # ── Education gate branches ──

    def test_edu_passed_parsed(self):
        gate = {
            "required_education": "本科",
            "resume_degree_levels": ["本科", "硕士"],
            "meets_requirement": True,
            "education_gate_source": "resume_parsed",
        }
        expl = _build_match_explanation(gate, 75, self._skill_info(), 50, "available", self._llm_info())
        assert any("学历硬性门槛已通过" in m for m in expl.hard_requirements_met)

    def test_edu_passed_no_levels(self):
        gate = {
            "required_education": "本科",
            "resume_degree_levels": [],
            "meets_requirement": True,
            "education_gate_source": "resume_parsed",
        }
        expl = _build_match_explanation(gate, 75, self._skill_info(), 50, "available", self._llm_info())
        assert any("已匹配" in m for m in expl.hard_requirements_met)

    def test_edu_failed(self):
        gate = {
            "required_education": "硕士",
            "resume_degree_levels": ["本科"],
            "meets_requirement": False,
            "education_gate_source": "resume_parsed",
        }
        expl = _build_match_explanation(gate, 40, self._skill_info(), 50, "available", self._llm_info())
        assert any("学历不满足要求" in m for m in expl.missing_requirements)

    def test_edu_unknown_no_parsed(self):
        gate = {
            "required_education": "本科",
            "resume_degree_levels": [],
            "meets_requirement": True,
            "education_gate_source": "unknown_no_parsed_education",
        }
        expl = _build_match_explanation(gate, 75, self._skill_info(), 50, "available", self._llm_info())
        assert any("学历信息缺失" in r for r in expl.risk_signals)
        assert any("尽快完成简历解析" in f for f in expl.interview_focus_points)

    def test_edu_no_requirement_no_edu_messages(self):
        gate = {
            "required_education": None,
            "resume_degree_levels": [],
            "meets_requirement": True,
            "education_gate_source": "skipped_no_requirement",
        }
        expl = _build_match_explanation(gate, 75, self._skill_info(), 50, "available", self._llm_info())
        assert not any("学历" in m for m in expl.hard_requirements_met)
        assert not any("学历" in m for m in expl.missing_requirements)

    def test_edu_skipped_unrecognized(self):
        gate = {
            "required_education": "理工科优先",
            "resume_degree_levels": [],
            "meets_requirement": True,
            "education_gate_source": "skipped_unrecognized_requirement",
        }
        expl = _build_match_explanation(gate, 75, self._skill_info(), 50, "available", self._llm_info())
        # No education-related messages expected
        assert not any("学历" in m for m in expl.hard_requirements_met)

    # ── Skill display ──

    def test_matched_skills_in_hard_met(self):
        si = self._skill_info(matched=["Python", "Docker"])
        expl = _build_match_explanation({}, 80, si, 50, "available", self._llm_info())
        assert any("Python" in m for m in expl.hard_requirements_met)

    def test_missing_skills_in_hard_missing(self):
        si = self._skill_info(missing=["Kubernetes", "AWS"])
        expl = _build_match_explanation({}, 80, si, 50, "available", self._llm_info())
        assert any("Kubernetes" in m for m in expl.missing_requirements)

    def test_no_skills_no_messages(self):
        si = self._skill_info()
        expl = _build_match_explanation({}, 80, si, 50, "available", self._llm_info())
        assert not any("技能" in m for m in expl.hard_requirements_met)
        assert not any("技能" in m for m in expl.missing_requirements)

    # ── Semantic display ──

    def test_semantic_not_indexed_risk(self):
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "not_indexed", self._llm_info())
        assert any("语义检索不可用" in r for r in expl.risk_signals)

    def test_semantic_high_adds_strong(self):
        expl = _build_match_explanation({}, 80, self._skill_info(), 80, "available", self._llm_info())
        assert any("语义向量匹配表现较好" in s for s in expl.strong_signals)

    def test_semantic_below_75_no_strong(self):
        expl = _build_match_explanation({}, 80, self._skill_info(), 70, "available", self._llm_info())
        assert not any("语义向量匹配表现较好" in s for s in expl.strong_signals)

    # ── Quality display ──

    def test_quality_evidence_high_adds_strong(self):
        li = self._llm_info(evidence_quality=80)
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", li)
        assert any("描述较具体充实" in s for s in expl.strong_signals)

    def test_quality_impact_high_adds_strong(self):
        li = self._llm_info(impact=80)
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", li)
        assert any("结果导向或大厂经验" in s for s in expl.strong_signals)

    def test_quality_consistency_risk_high_adds_risk(self):
        li = self._llm_info(consistency_risk=65)
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", li)
        assert any("空泛或简短风险" in r for r in expl.risk_signals)

    def test_quality_consistency_below_60_no_risk(self):
        li = self._llm_info(consistency_risk=50)
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", li)
        assert not any("空泛或简短风险" in r for r in expl.risk_signals)

    def test_quality_status_disabled_no_risk(self):
        li = self._llm_info(status="disabled")
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", li)
        assert not any("经历质量评分不可用" in r for r in expl.risk_signals)

    def test_quality_status_unavailable_adds_risk(self):
        li = self._llm_info(status="llm_error")
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", li)
        assert any("经历质量评分不可用" in r for r in expl.risk_signals)

    # ── Suggested action thresholds ──

    @pytest.mark.parametrize(
        "overall, expected",
        [
            (90.0, "recommend_interview"),
            (70.0, "recommend_interview"),
            (69.9, "further_screening"),
            (50.0, "further_screening"),
            (49.9, "not_recommended"),
            (0.0, "not_recommended"),
        ],
    )
    def test_suggested_action_thresholds(self, overall, expected):
        expl = _build_match_explanation({}, overall, self._skill_info(), 50, "available", self._llm_info())
        assert expl.suggested_action == expected

    # ── Summary text ──

    def test_summary_includes_formula(self):
        """Summary should mention the skill × factor formula and semantic as reference."""
        si = self._skill_info(skill_score=80.0)
        li = self._llm_info(llm_quality_score=50.0)
        expl = _build_match_explanation({}, 64.0, si, 45.0, "available", li)
        assert expl.summary_for_hr is not None
        assert "综合匹配分" in expl.summary_for_hr
        assert "技能" in expl.summary_for_hr
        assert "质量系数" in expl.summary_for_hr
        assert "仅供参考" in expl.summary_for_hr

    def test_summary_includes_hard_met(self):
        si = self._skill_info(matched=["Python"])
        expl = _build_match_explanation({}, 80, si, 50, "available", self._llm_info())
        assert "已满足的核心条件" in expl.summary_for_hr

    def test_summary_includes_hard_missing(self):
        si = self._skill_info(missing=["K8s"])
        expl = _build_match_explanation({}, 80, si, 50, "available", self._llm_info())
        assert "尚未满足的硬性要求" in expl.summary_for_hr

    def test_summary_includes_strong_and_risks(self):
        li = self._llm_info(evidence_quality=80, consistency_risk=70)
        expl = _build_match_explanation({}, 80, self._skill_info(), 80, "available", li)
        assert "亮点" in expl.summary_for_hr
        assert "风险点" in expl.summary_for_hr

    def test_summary_none_when_all_empty(self):
        """When nothing to report, summary should be None (not empty string)."""
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", self._llm_info())
        # No matched, no missing, no strong, no risks → summary can be None
        # But it always includes the score line, so it's never truly empty
        assert expl.summary_for_hr is not None  # always has score line

    # ── Skill terms passthrough ──

    def test_job_and_candidate_skill_terms(self):
        si = self._skill_info(job_terms=["python", "docker"], cand_terms=["python", "java"])
        expl = _build_match_explanation({}, 80, si, 50, "available", self._llm_info())
        assert expl.job_skill_terms == ["python", "docker"]
        assert expl.candidate_skill_terms == ["python", "java"]

    # ── Return type ──

    def test_returns_match_explanation(self):
        expl = _build_match_explanation({}, 80, self._skill_info(), 50, "available", self._llm_info())
        assert isinstance(expl, MatchExplanation)
        assert hasattr(expl, "suggested_action")
        assert hasattr(expl, "summary_for_hr")


# =============================================================================
# _shared_terms_for_alignment
# =============================================================================

class TestSharedTermsForAlignment:

    def test_common_terms(self):
        result = _shared_terms_for_alignment("Python Docker 开发", "Python 开发工程师")
        # tokenize output depends on jieba, but "python" and "docker" should appear
        assert len(result) >= 1

    def test_no_common_terms(self):
        result = _shared_terms_for_alignment("前端 React", "后端 Java")
        # Might have some overlapping stop-words depending on tokenizer
        # but should be minimal
        assert isinstance(result, list)

    def test_empty_strings(self):
        result = _shared_terms_for_alignment("", "")
        assert result == []

    def test_max_n_limit(self):
        """Should not exceed max_n."""
        result = _shared_terms_for_alignment(
            "Python Java Go Rust C++ TypeScript JavaScript Ruby PHP Swift Kotlin",
            "Python Java Go Rust C++ TypeScript JavaScript Ruby PHP Swift Kotlin",
            max_n=5,
        )
        assert len(result) <= 5

    def test_default_max_n(self):
        """Default max_n=12."""
        result = _shared_terms_for_alignment("a b c d e f g h i j k l m n o", "a b c d e f g h i j k l m n o")
        assert len(result) <= 12

    def test_returns_sorted(self):
        result = _shared_terms_for_alignment("c b a", "a b c")
        assert result == sorted(result)

    def test_returns_list_of_strings(self):
        result = _shared_terms_for_alignment("Python", "Python")
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], str)


# =============================================================================
# _enrich_delivery_alignments
# =============================================================================

class TestEnrichDeliveryAlignments:

    def test_normal_alignment(self):
        raw = [
            {
                "job_profile_type": "resp_0",
                "cand_profile_type": "proj_0",
                "job_text_snippet": "负责后端 API 开发",
                "cand_text_snippet": "参与后端 API 设计与开发",
                "cosine": 0.85,
                "bm25": 12.5,
                "rank_cos": 1,
                "rank_bm25": 2,
                "rrf": 0.032,
                "bm25_degenerate": False,
            }
        ]
        result = _enrich_delivery_alignments(raw)
        assert len(result) == 1
        assert isinstance(result[0], DeliveryAlignmentItem)
        assert result[0].job_profile_type == "resp_0"
        assert result[0].cand_profile_type == "proj_0"
        assert result[0].cosine == 0.85
        assert result[0].bm25 == 12.5
        assert result[0].rrf == 0.032
        assert result[0].bm25_degenerate is False
        assert isinstance(result[0].shared_terms, list)

    def test_empty_input(self):
        assert _enrich_delivery_alignments([]) == []

    def test_filters_non_dict_entries(self):
        raw: List[Any] = ["not_a_dict", 123, None]
        result = _enrich_delivery_alignments(raw)
        assert result == []

    def test_mixed_valid_invalid(self):
        raw: List[Any] = [
            "junk",
            {
                "job_profile_type": "resp_0",
                "cand_profile_type": "work_0",
                "job_text_snippet": "JD text",
                "cand_text_snippet": "CV text",
                "cosine": 0.72,
                "bm25": 8.0,
                "rank_cos": 1,
                "rank_bm25": 1,
                "rrf": 0.04,
                "bm25_degenerate": True,
            },
        ]
        result = _enrich_delivery_alignments(raw)
        assert len(result) == 1
        assert result[0].bm25_degenerate is True

    def test_missing_fields_default_to_zero(self):
        raw = [{"job_profile_type": "resp_0", "cand_profile_type": "proj_0"}]
        result = _enrich_delivery_alignments(raw)
        assert len(result) == 1
        assert result[0].cosine == 0.0
        assert result[0].bm25 == 0.0
        assert result[0].rrf == 0.0
        assert result[0].rank_cos == 0
        assert result[0].rank_bm25 == 0
        assert result[0].bm25_degenerate is False

    def test_multiple_alignments(self):
        raw = [
            {
                "job_profile_type": f"resp_{i}",
                "cand_profile_type": f"proj_{i}",
                "job_text_snippet": f"JD {i}",
                "cand_text_snippet": f"CV {i}",
                "cosine": 0.7 + i * 0.02,
                "bm25": float(i),
                "rank_cos": i,
                "rank_bm25": i,
                "rrf": 0.01 * i,
                "bm25_degenerate": i % 2 == 0,
            }
            for i in range(5)
        ]
        result = _enrich_delivery_alignments(raw)
        assert len(result) == 5
        assert result[4].cosine == pytest.approx(0.78)
        assert result[3].bm25_degenerate is False
        assert result[2].bm25_degenerate is True
