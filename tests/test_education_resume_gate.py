"""Unit tests for education_resume_gate — degree ranking and education hard filter."""

import uuid
from unittest.mock import MagicMock

import pytest

from app import statuses
from app.database.models import CandidateModel, JobModel, ResumeModel
from app.services.education_resume_gate import (
    RANK_BACHELOR,
    RANK_COLLEGE,
    RANK_DOCTOR,
    RANK_MASTER,
    _entries_pass_requirement,
    _evaluate_resumes_for_requirement,
    _max_degree_rank_from_entries,
    _nfkc_lower,
    degree_levels_from_entries,
    degree_rank_from_text,
    education_entries_from_parsed,
    job_education_requirement_text,
    job_minimum_degree_rank,
)


# ===================================================================
# _nfkc_lower
# ===================================================================
def test_nfkc_lower_fullwidth():
    """Fullwidth Latin letters should normalize to ASCII."""
    # Fullwidth 'A' (U+FF21) → 'a'
    assert _nfkc_lower("Ａ") == "a"


def test_nfkc_lower_empty():
    assert _nfkc_lower("") == ""
    # None → or "" fallback in the function gives ""
    assert _nfkc_lower(None) == ""


def test_nfkc_lower_chinese_unchanged():
    """Chinese characters should not be mangled by NFKC."""
    assert _nfkc_lower("本科") == "本科"
    assert _nfkc_lower("博士") == "博士"


# ===================================================================
# degree_rank_from_text
# ===================================================================
class TestDegreeRankFromText:
    def test_doctor(self):
        assert degree_rank_from_text("博士") == RANK_DOCTOR
        assert degree_rank_from_text("博士学位") == RANK_DOCTOR

    def test_master(self):
        assert degree_rank_from_text("硕士") == RANK_MASTER
        assert degree_rank_from_text("硕士研究生") == RANK_MASTER

    def test_bachelor(self):
        assert degree_rank_from_text("本科") == RANK_BACHELOR
        assert degree_rank_from_text("大学本科") == RANK_BACHELOR

    def test_college(self):
        assert degree_rank_from_text("大专") == RANK_COLLEGE

    def test_case_insensitive(self):
        """Degree text matching is case-insensitive (NFKC lower)."""
        assert degree_rank_from_text("硕士") == RANK_MASTER
        assert degree_rank_from_text("博士") == RANK_DOCTOR

    def test_empty_and_none(self):
        assert degree_rank_from_text("") is None
        assert degree_rank_from_text("   ") is None

    def test_garbage_text(self):
        assert degree_rank_from_text("高中") is None
        assert degree_rank_from_text("MBA") is None
        assert degree_rank_from_text("随便写的") is None

    def test_highest_match_wins_if_multiple_keywords(self):
        """If text contains multiple degree keywords, the FIRST matched
        in order 博士→硕士→本科→大专 wins. 硕士 is checked before 本科."""
        text = "本科毕业，在职硕士在读"
        # 硕士 found first (before 本科 in check order) → RANK_MASTER
        assert degree_rank_from_text(text) == RANK_MASTER

    def test_master_found_before_bachelor(self):
        # "硕士" is checked before "本科", so "本科毕业在职硕士" should return 硕士
        assert degree_rank_from_text("本科毕业在职硕士") == RANK_MASTER

    def test_bachelor_found_before_college(self):
        assert degree_rank_from_text("大专升本科") == RANK_BACHELOR


# ===================================================================
# job_minimum_degree_rank
# ===================================================================
class TestJobMinimumDegreeRank:
    def test_doctor_requirement(self):
        assert job_minimum_degree_rank("博士及以上") == RANK_DOCTOR

    def test_master_requirement(self):
        assert job_minimum_degree_rank("硕士及以上学历") == RANK_MASTER
        assert job_minimum_degree_rank("硕士研究生") == RANK_MASTER

    def test_bachelor_requirement(self):
        assert job_minimum_degree_rank("本科及以上") == RANK_BACHELOR
        assert job_minimum_degree_rank("本科学历") == RANK_BACHELOR

    def test_college_requirement(self):
        assert job_minimum_degree_rank("大专及以上") == RANK_COLLEGE

    def test_empty_and_none(self):
        assert job_minimum_degree_rank("") is None
        assert job_minimum_degree_rank("   ") is None

    def test_unrecognized(self):
        """Unrecognized requirement → None (gate not enforced)."""
        assert job_minimum_degree_rank("学历不限") is None
        assert job_minimum_degree_rank("面议") is None


# ===================================================================
# job_education_requirement_text
# ===================================================================
class TestJobEducationRequirementText:
    def test_from_job_direct_field(self):
        job = JobModel(title="测试", education_requirement="本科及以上")
        assert job_education_requirement_text(job, None) == "本科及以上"

    def test_from_structured_fallback(self):
        job = JobModel(title="测试", education_requirement=None)
        structured = {"education_requirement": "硕士及以上学历"}
        assert job_education_requirement_text(job, structured) == "硕士及以上学历"

    def test_direct_field_priority_over_structured(self):
        job = JobModel(title="测试", education_requirement="本科")
        structured = {"education_requirement": "硕士"}
        # Direct field wins
        assert job_education_requirement_text(job, structured) == "本科"

    def test_both_none(self):
        job = JobModel(title="测试", education_requirement=None)
        assert job_education_requirement_text(job, None) is None

    def test_structured_missing_key(self):
        job = JobModel(title="测试", education_requirement=None)
        structured = {"skills": "Python"}
        assert job_education_requirement_text(job, structured) is None


# ===================================================================
# education_entries_from_parsed
# ===================================================================
class TestEducationEntriesFromParsed:
    def test_from_layer_1_extracted(self):
        parsed = {
            "layer_1_extracted": {
                "education": [
                    {"degree": "本科", "school": "清华大学"},
                    {"degree": "硕士", "school": "北京大学"},
                ]
            }
        }
        entries = education_entries_from_parsed(parsed)
        assert len(entries) == 2
        assert entries[0]["degree"] == "本科"
        assert entries[1]["degree"] == "硕士"

    def test_fallback_to_top_level_education(self):
        parsed = {"education": [{"degree": "博士", "school": "中科院"}]}
        entries = education_entries_from_parsed(parsed)
        assert len(entries) == 1
        assert entries[0]["degree"] == "博士"

    def test_layer_1_takes_priority(self):
        """layer_1_extracted.education is checked first."""
        parsed = {
            "layer_1_extracted": {"education": [{"degree": "硕士"}]},
            "education": [{"degree": "本科"}],
        }
        entries = education_entries_from_parsed(parsed)
        assert len(entries) == 1
        assert entries[0]["degree"] == "硕士"

    def test_none_parsed(self):
        assert education_entries_from_parsed(None) == []

    def test_non_dict_parsed(self):
        assert education_entries_from_parsed("not a dict") == []

    def test_empty_parsed(self):
        assert education_entries_from_parsed({}) == []

    def test_layer_1_exists_but_no_education_field(self):
        parsed = {"layer_1_extracted": {"skills": ["Python"]}}
        entries = education_entries_from_parsed(parsed)
        assert entries == []

    def test_education_not_a_list(self):
        parsed = {"layer_1_extracted": {"education": "not a list"}}
        entries = education_entries_from_parsed(parsed)
        assert entries == []

    def test_filters_non_dict_entries(self):
        parsed = {
            "layer_1_extracted": {
                "education": [
                    {"degree": "本科"},
                    "not a dict",
                    {"degree": "硕士"},
                    None,
                ]
            }
        }
        entries = education_entries_from_parsed(parsed)
        assert len(entries) == 2


# ===================================================================
# degree_levels_from_entries
# ===================================================================
class TestDegreeLevelsFromEntries:
    def test_extracts_degree_strings(self):
        entries = [
            {"degree": "本科", "school": "清华"},
            {"degree": "硕士", "school": "北大"},
        ]
        assert degree_levels_from_entries(entries) == ["本科", "硕士"]

    def test_missing_degree_field(self):
        entries = [{"school": "清华"}, {"degree": ""}]
        assert degree_levels_from_entries(entries) == []

    def test_empty_entries(self):
        assert degree_levels_from_entries([]) == []


# ===================================================================
# _max_degree_rank_from_entries
# ===================================================================
class TestMaxDegreeRankFromEntries:
    def test_picks_highest(self):
        entries = [
            {"degree": "本科"},
            {"degree": "博士"},
            {"degree": "硕士"},
        ]
        assert _max_degree_rank_from_entries(entries) == RANK_DOCTOR

    def test_single_entry(self):
        assert _max_degree_rank_from_entries([{"degree": "本科"}]) == RANK_BACHELOR

    def test_all_unrecognized(self):
        entries = [{"degree": "高中"}, {"degree": "MBA"}]
        assert _max_degree_rank_from_entries(entries) is None

    def test_some_recognized_some_not(self):
        entries = [{"degree": "高中"}, {"degree": "硕士"}]
        assert _max_degree_rank_from_entries(entries) == RANK_MASTER

    def test_empty(self):
        assert _max_degree_rank_from_entries([]) is None


# ===================================================================
# _entries_pass_requirement
# ===================================================================
class TestEntriesPassRequirement:
    def test_bachelor_meets_bachelor(self):
        entries = [{"degree": "本科"}]
        ok, levels, best = _entries_pass_requirement(RANK_BACHELOR, entries)
        assert ok is True
        assert levels == ["本科"]
        assert best == RANK_BACHELOR

    def test_master_exceeds_bachelor(self):
        entries = [{"degree": "硕士"}]
        ok, levels, best = _entries_pass_requirement(RANK_BACHELOR, entries)
        assert ok is True
        assert best == RANK_MASTER

    def test_college_fails_bachelor(self):
        entries = [{"degree": "大专"}]
        ok, levels, best = _entries_pass_requirement(RANK_BACHELOR, entries)
        assert ok is False
        assert best == RANK_COLLEGE

    def test_no_recognized_degree_fails(self):
        entries = [{"degree": "高中"}]
        ok, levels, best = _entries_pass_requirement(RANK_BACHELOR, entries)
        assert ok is False
        assert best is None

    def test_empty_entries_fails(self):
        ok, levels, best = _entries_pass_requirement(RANK_BACHELOR, [])
        assert ok is False


# ===================================================================
# _evaluate_resumes_for_requirement
# ===================================================================
class TestEvaluateResumesForRequirement:
    @staticmethod
    def _resume(parsed=None, status=statuses.RESUME_STATUS_PARSED):
        """Helper to build a ResumeModel."""
        r = ResumeModel(
            candidate_id=uuid.uuid4(),
            status=status,
        )
        r.parsed = parsed
        return r

    def test_single_resume_passes(self):
        resumes = [
            self._resume(
                parsed={"layer_1_extracted": {"education": [{"degree": "硕士"}]}}
            )
        ]
        ok, levels, best = _evaluate_resumes_for_requirement(RANK_BACHELOR, resumes)
        assert ok is True
        assert "硕士" in levels
        assert best == RANK_MASTER

    def test_single_resume_fails(self):
        resumes = [
            self._resume(
                parsed={"layer_1_extracted": {"education": [{"degree": "大专"}]}}
            )
        ]
        ok, levels, best = _evaluate_resumes_for_requirement(RANK_BACHELOR, resumes)
        assert ok is False
        # When no resume passes, the function returns (False, [], None)
        # — best rank from failing entries is discarded
        assert best is None
        assert levels == []

    def test_first_resume_no_parsed_second_passes(self):
        """If first resume has no parsed data, skip it and check the next."""
        resumes = [
            self._resume(parsed=None),
            self._resume(
                parsed={"layer_1_extracted": {"education": [{"degree": "本科"}]}}
            ),
        ]
        ok, _, best = _evaluate_resumes_for_requirement(RANK_BACHELOR, resumes)
        assert ok is True
        assert best == RANK_BACHELOR

    def test_skips_non_usable_status(self):
        """Resumes with status != parsed/candidate_bound are skipped."""
        resumes = [
            self._resume(
                parsed={"layer_1_extracted": {"education": [{"degree": "本科"}]}},
                status=statuses.RESUME_STATUS_UPLOADED,  # not usable
            ),
        ]
        ok, levels, best = _evaluate_resumes_for_requirement(RANK_BACHELOR, resumes)
        # Should skip the uploaded resume → no data → fail
        assert ok is False

    def test_candidate_bound_status_is_usable(self):
        """candidate_bound is also a usable status."""
        resumes = [
            self._resume(
                parsed={"layer_1_extracted": {"education": [{"degree": "博士"}]}},
                status=statuses.RESUME_STATUS_CANDIDATE_BOUND,
            ),
        ]
        ok, _, best = _evaluate_resumes_for_requirement(RANK_BACHELOR, resumes)
        assert ok is True
        assert best == RANK_DOCTOR

    def test_all_resumes_no_parsed(self):
        resumes = [self._resume(parsed=None), self._resume(parsed=None)]
        ok, levels, best = _evaluate_resumes_for_requirement(RANK_BACHELOR, resumes)
        assert ok is False


# ===================================================================
# rank ordering sanity
# ===================================================================
def test_rank_ordering():
    """Verify the rank integers form a proper hierarchy."""
    assert RANK_DOCTOR > RANK_MASTER > RANK_BACHELOR > RANK_COLLEGE

    # Verify specific values
    assert RANK_COLLEGE == 0
    assert RANK_BACHELOR == 1
    assert RANK_MASTER == 2
    assert RANK_DOCTOR == 3
