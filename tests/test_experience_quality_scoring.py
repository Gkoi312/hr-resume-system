"""Unit tests for rule-based experience quality scoring."""

from unittest.mock import MagicMock

from app.database.models import CandidateModel
from app.services.experience_quality_scoring import (
    _all_descriptions,
    _best_company_score,
    _company_score,
    _description_specificity_score,
    score_candidate_experience_quality,
)


# ---------------------------------------------------------------------------
# _company_score
# ---------------------------------------------------------------------------
def test_company_score_none_or_empty():
    assert _company_score(None) == 0.0
    assert _company_score("") == 0.0
    assert _company_score("   ") == 0.0


def test_company_score_exact_match():
    assert _company_score("腾讯") == 10.0
    assert _company_score("Google") == 10.0
    assert _company_score("美团") == 9.0
    assert _company_score("shopee") == 7.0
    assert _company_score("中金") == 6.0


def test_company_score_substring_match():
    # "腾讯科技" contains "腾讯"
    assert _company_score("腾讯科技") == 10.0
    # "字节跳动" is exact
    assert _company_score("字节跳动") == 10.0


def test_company_score_unrecognized():
    # Unknown company → 2.0 baseline
    assert _company_score("某不知名小公司") == 2.0


# ---------------------------------------------------------------------------
# _best_company_score
# ---------------------------------------------------------------------------
def test_best_company_score_empty():
    assert _best_company_score([]) == 0.0


def test_best_company_score_picks_highest():
    exps = [
        {"company": "某小厂"},
        {"company": "腾讯"},
        {"company": "美团"},  # tier 9, lower than 腾讯's 10
    ]
    assert _best_company_score(exps) == 10.0


# ---------------------------------------------------------------------------
# _description_specificity_score
# ---------------------------------------------------------------------------
def test_specificity_empty_description():
    result = _description_specificity_score(None)
    assert result["length"] == 0
    assert result["has_metrics"] is False
    assert result["has_action_result"] is False


def test_specificity_short_vague():
    # "做了一些事情" has zero specificity signals — no metrics, no action patterns
    result = _description_specificity_score("做了一些事情")
    assert result["length"] > 0
    assert result["has_metrics"] is False
    assert result["has_action_result"] is False


def test_specificity_with_metrics():
    desc = "使用Python开发XX系统，日处理100万请求，QPS从500提升到2000，延迟降低了40%"
    result = _description_specificity_score(desc)
    assert result["has_metrics"] is True
    assert result["metric_count"] >= 2  # 100万, 500, 2000, 40%


def test_specificity_with_action_result():
    desc = "负责设计和开发了分布式消息引擎，通过异步架构将系统吞吐量提升了3倍"
    result = _description_specificity_score(desc)
    assert result["has_action_result"] is True


def test_specificity_with_both():
    desc = (
        "主导搭建了用户画像平台，覆盖5000万用户，通过引入Flink实时计算框架，"
        "将数据延迟从小时级降低到秒级，日均处理数据量10TB+"
    )
    result = _description_specificity_score(desc)
    assert result["has_metrics"] is True
    assert result["has_action_result"] is True


# ---------------------------------------------------------------------------
# _all_descriptions
# ---------------------------------------------------------------------------
def test_all_descriptions_empty():
    assert _all_descriptions([], []) == []


def test_all_descriptions_extracts_all():
    work = [{"description": "desc1"}, {"title": "no desc field"}]
    proj = [{"description": "desc2"}]
    result = _all_descriptions(work, proj)
    assert result == ["desc1", "desc2"]


# ---------------------------------------------------------------------------
# score_candidate_experience_quality — integration
# ---------------------------------------------------------------------------
def _make_candidate(
    work_experience=None,
    projects=None,
    skills=None,
    summary=None,
):
    """Helper to build a CandidateModel-like object."""
    c = MagicMock(spec=CandidateModel)
    c.work_experience = work_experience or []
    c.projects = projects or []
    c.skills = skills or []
    c.summary = summary
    return c


def test_score_empty_candidate():
    c = _make_candidate()
    result = score_candidate_experience_quality(c)
    assert result["status"] == "rule_based"
    assert result["llm_quality_score"] == 0.0
    assert "缺少实习和项目描述" in result["summary"]


def test_score_vague_candidate():
    c = _make_candidate(
        work_experience=[
            {"company": "某小公司", "title": "实习生", "description": "负责日常开发工作"},
        ],
        projects=[
            {"name": "课程项目", "description": "完成了一个管理系统"},
        ],
        skills=["Python"],
    )
    result = score_candidate_experience_quality(c)
    # Should get some score (not zero) since there IS content, just vague
    assert result["llm_quality_score"] > 0
    assert result["llm_quality_score"] < 50  # Should be low due to vagueness
    assert "缺少量化数据" in result["summary"] or "偏简短" in result["summary"]


def test_score_strong_candidate():
    c = _make_candidate(
        work_experience=[
            {
                "company": "腾讯科技",
                "title": "后端开发实习生",
                "description": (
                    "负责微信支付核心系统的开发和维护，设计并实现了分布式事务方案，"
                    "日均处理交易量5000万笔，通过优化数据库索引和缓存策略，"
                    "将查询延迟从200ms降低到50ms，系统可用性从99.9%提升到99.99%。"
                    "独立设计了资金对账的T+0实时核对系统，覆盖12个业务场景，"
                    "减少了人工对账成本约200人天/年。编写了核心模块的技术文档和"
                    "故障复盘手册，参与Code Review超过200次。"
                ),
            },
            {
                "company": "美团",
                "title": "算法实习生",
                "description": (
                    "参与推荐系统排序模型优化，通过引入多任务学习框架，"
                    "CTR提升了3.2%，用户停留时长增加8%。负责特征工程pipeline"
                    "的重构和优化，将特征计算延迟从小时级降低到分钟级，"
                    "日均处理特征量5000万+。与产品、运营团队协作完成了3次"
                    "AB实验的设计、上线和效果分析，推动模型迭代落地。"
                ),
            },
        ],
        projects=[
            {
                "name": "分布式KV存储",
                "role": "核心开发者",
                "description": (
                    "基于Raft协议实现了强一致性的分布式KV存储引擎，"
                    "支持10万+QPS读写，延迟P99<10ms，已开源获得2000+ star。"
                    "设计了快照备份和增量恢复机制，支持PB级数据规模下的高效迁移，"
                    "写入了30+篇设计文档和API文档。在GitHub上维护项目社区，"
                    "处理Issue和PR超过150个，项目被多家公司引入生产环境使用。"
                ),
            },
        ],
        skills=["Go", "Python", "Redis", "Kubernetes", "gRPC", "MySQL"],
        summary="5年后端开发经验，擅长分布式系统和性能优化",
    )
    result = score_candidate_experience_quality(c)
    assert result["status"] == "rule_based"
    assert result["llm_quality_score"] >= 65  # 及格线200字 + 高具体性 + 大厂
    assert "大厂实习" in result["summary"]


def test_score_only_skills_no_experience():
    c = _make_candidate(
        skills=["Python", "Django", "React"],
    )
    result = score_candidate_experience_quality(c)
    assert result["llm_quality_score"] > 0  # Has skills listed
    assert result["llm_quality_score"] < 30  # But no experience or projects


def test_score_many_projects_no_work():
    c = _make_candidate(
        projects=[
            {"name": f"项目{i}", "description": f"项目{i}的详细描述，使用多种技术栈，完成了XX功能模块的开发与测试"}
            for i in range(5)
        ],
        skills=["Java", "Spring Boot", "MySQL"],
    )
    result = score_candidate_experience_quality(c)
    assert result["llm_quality_score"] > 0
    # Having 5 projects should give decent breadth score
    assert result["evidence_quality_score"] > 0


# ---------------------------------------------------------------------------
# score_candidate_experience_quality — return structure compatibility
# ---------------------------------------------------------------------------
def test_return_structure_matches_llm_format():
    """The rule-based scorer must return keys compatible with LLM scorer."""
    c = _make_candidate(
        work_experience=[{"company": "华为", "title": "开发", "description": "参与5G基站软件开发"}],
        skills=["C++"],
    )
    result = score_candidate_experience_quality(c)
    required_keys = [
        "impact_score",
        "evidence_quality_score",
        "consistency_risk",
        "llm_quality_score",
        "summary",
        "status",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    # All numeric fields should be in 0-100 range
    for numeric_key in ["impact_score", "evidence_quality_score", "consistency_risk", "llm_quality_score"]:
        assert 0.0 <= result[numeric_key] <= 100.0, f"{numeric_key} out of range: {result[numeric_key]}"
