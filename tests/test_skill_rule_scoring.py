"""Unit tests for skill_rule_scoring — rule-based skill matching."""

import pytest

from app.database.models import CandidateModel, JobModel
from app.services.skill_rule_scoring import _clean_terms, score_candidate_skills


# ---------------------------------------------------------------------------
# helpers to build minimal ORM objects (no DB)
# ---------------------------------------------------------------------------

def _job(**kw) -> JobModel:
    return JobModel(
        title=kw.get("title", "测试岗位"),
        structured=kw.get("structured", {}),
    )


def _candidate(**kw) -> CandidateModel:
    return CandidateModel(
        name=kw.get("name", "测试"),
        email=kw.get("email", "test@example.com"),
        skills=kw.get("skills", []),
    )


# ---------------------------------------------------------------------------
# _clean_terms
# ---------------------------------------------------------------------------

def test_clean_terms_normal():
    # Redis and Kafka each have their own canonical (no alias collision)
    assert _clean_terms(["Redis", "Kafka"]) == ["redis", "kafka"]


def test_clean_terms_alias_dedup():
    # k8s and kubernetes share the same canonical → deduped to one
    assert _clean_terms(["k8s", "kubernetes"]) == ["kubernetes"]


def test_clean_terms_dedup():
    assert _clean_terms(["Python", "python", "PYTHON"]) == ["python"]


def test_clean_terms_empty_list():
    assert _clean_terms([]) == []


def test_clean_terms_not_a_list():
    assert _clean_terms(None) == []
    assert _clean_terms("not_a_list") == []
    assert _clean_terms(123) == []


def test_clean_terms_strips_whitespace():
    assert _clean_terms(["  Redis  ", "\tKafka\n"]) == ["redis", "kafka"]


def test_clean_terms_skips_empty_strings():
    assert _clean_terms(["Redis", "", "  ", "Kafka"]) == ["redis", "kafka"]


# ---------------------------------------------------------------------------
# score_candidate_skills — basic matching
# ---------------------------------------------------------------------------

def test_perfect_match():
    # Redis, Kafka, MongoDB — each has a distinct canonical, no alias collision
    job = _job(structured={"required_skills": ["Redis", "Kafka", "MongoDB"]})
    cand = _candidate(skills=["Redis", "Kafka", "MongoDB"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0
    assert set(r["matched_skills"]) == {"redis", "kafka", "mongodb"}
    assert r["missing_skills"] == []
    assert r["status"] == "available"


def test_partial_match():
    job = _job(structured={"required_skills": ["Redis", "Kafka", "Docker"]})
    cand = _candidate(skills=["Redis"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == pytest.approx(33.3, abs=0.1)
    assert r["matched_skills"] == ["redis"]
    assert set(r["missing_skills"]) == {"kafka", "docker"}
    assert r["status"] == "available"


def test_zero_match():
    job = _job(structured={"required_skills": ["Redis", "Kafka"]})
    cand = _candidate(skills=["Java", "Spring"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 0.0
    assert r["matched_skills"] == []
    assert set(r["missing_skills"]) == {"redis", "kafka"}


def test_no_job_skills():
    job = _job(structured={})
    cand = _candidate(skills=["Python"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 0.0
    assert r["status"] == "no_job_skills"
    assert r["job_skill_terms"] == []


def test_no_candidate_skills():
    job = _job(structured={"required_skills": ["Python"]})
    cand = _candidate(skills=[])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 0.0
    assert r["status"] == "no_candidate_skills"
    assert r["matched_skills"] == []
    assert set(r["missing_skills"]) == {"python"}


def test_candidate_skills_none():
    job = _job(structured={"required_skills": ["Python"]})
    cand = _candidate(skills=None)
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 0.0
    assert r["status"] == "no_candidate_skills"


# ---------------------------------------------------------------------------
# preferred_skills
# ---------------------------------------------------------------------------

def test_preferred_skills_merged():
    job = _job(structured={
        "required_skills": ["Python"],
        "preferred_skills": ["Docker", "Redis"],
    })
    cand = _candidate(skills=["python", "redis"])
    r = score_candidate_skills(job, cand)
    # job terms: python, docker, redis (3)
    # matched: python, redis (2)
    assert r["skill_score"] == pytest.approx(66.7, abs=0.1)
    assert set(r["matched_skills"]) == {"python", "redis"}
    assert r["missing_skills"] == ["docker"]


def test_preferred_duplicate_with_required_does_not_double_count():
    job = _job(structured={
        "required_skills": ["Python"],
        "preferred_skills": ["Python"],
    })
    cand = _candidate(skills=["python"])
    r = score_candidate_skills(job, cand)
    # job terms should be ["python"] not ["python", "python"]
    assert r["job_skill_terms"] == ["python"]
    assert r["skill_score"] == 100.0


# ---------------------------------------------------------------------------
# alias / canonical mapping (via skill_evidence lexicon)
# ---------------------------------------------------------------------------

def test_alias_golang_maps_to_go():
    job = _job(structured={"required_skills": ["go"]})
    cand = _candidate(skills=["golang"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0
    assert "go" in r["matched_skills"]


def test_alias_k8s_maps_to_kubernetes():
    job = _job(structured={"required_skills": ["kubernetes"]})
    cand = _candidate(skills=["k8s"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0


def test_alias_springboot_maps_to_spring_boot():
    job = _job(structured={"required_skills": ["spring boot"]})
    cand = _candidate(skills=["SpringBoot"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0


def test_chinese_skill_machine_learning():
    job = _job(structured={"required_skills": ["机器学习"]})
    cand = _candidate(skills=["machine learning"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0


def test_django_is_not_python():
    """After fix: django should stay django, NOT be collapsed into python."""
    job = _job(structured={"required_skills": ["Python"]})
    cand = _candidate(skills=["django"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 0.0
    assert "django" not in r["matched_skills"]


def test_vue_and_react_are_distinct():
    """After fix: vue and react each have their own canonical, not merged to javascript."""
    job = _job(structured={"required_skills": ["vue", "react"]})
    cand = _candidate(skills=["vue", "react"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0
    assert "vue" in r["matched_skills"]
    assert "react" in r["matched_skills"]


def test_postgresql_is_not_sql():
    """After fix: postgresql should stay postgresql, not collapsed into sql."""
    job = _job(structured={"required_skills": ["sql"]})
    cand = _candidate(skills=["postgresql"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 0.0
    assert "postgresql" not in r["matched_skills"]


def test_alias_not_in_lexicon_passed_through():
    """Unknown skill terms keep their original (lowercased) form."""
    job = _job(structured={"required_skills": ["SomeObscureTool"]})
    cand = _candidate(skills=["SomeObscureTool"])
    r = score_candidate_skills(job, cand)
    # both sides normalize to "someobscuretool", should match
    assert r["skill_score"] == 100.0


# ---------------------------------------------------------------------------
# case insensitivity & whitespace
# ---------------------------------------------------------------------------

def test_case_insensitive():
    job = _job(structured={"required_skills": ["PYTHON", "FastApi"]})
    cand = _candidate(skills=["python", "FASTAPI"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0


def test_whitespace_in_candidate_skills():
    job = _job(structured={"required_skills": ["Python", "SQL"]})
    cand = _candidate(skills=["  Python  ", "\tSQL\n"])
    r = score_candidate_skills(job, cand)
    assert r["skill_score"] == 100.0
