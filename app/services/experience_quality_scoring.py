"""Rule-based experience quality scoring.

Evaluates candidate experience quality the way HR would:
- Are descriptions detailed or empty/placeholder?
- How many experiences/projects does the candidate have?
- Are there concrete signals (numbers, tools, outcomes)?
- Does the candidate have experience at recognized companies?

No LLM calls — pure rule-based, fast, deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.database.models import CandidateModel

# ---------------------------------------------------------------------------
# Recognized companies — HR notices these immediately
# ---------------------------------------------------------------------------
_RECOGNIZED_COMPANIES: Dict[str, int] = {
    # Tier 1 — global tech giants
    "google": 10,
    "microsoft": 10,
    "apple": 10,
    "amazon": 10,
    "meta": 10,
    "facebook": 10,
    "netflix": 9,
    "nvidia": 9,
    "openai": 9,
    "anthropic": 9,
    # Tier 1 — Chinese tech giants
    "腾讯": 10,
    "tencent": 10,
    "阿里巴巴": 10,
    "alibaba": 10,
    "字节跳动": 10,
    "bytedance": 10,
    "抖音": 9,
    "tiktok": 9,
    "百度": 9,
    "baidu": 9,
    "华为": 9,
    "huawei": 9,
    "美团": 9,
    "meituan": 9,
    "滴滴": 8,
    "didi": 8,
    "京东": 8,
    "jd.com": 8,
    "拼多多": 8,
    "pinduoduo": 8,
    "网易": 8,
    "netease": 8,
    "快手": 8,
    "kuaishou": 8,
    "小米": 8,
    "xiaomi": 8,
    "蚂蚁": 8,
    "ant group": 8,
    "蚂蚁集团": 8,
    # Tier 2 — notable tech companies
    "shopee": 7,
    "grab": 7,
    "gojek": 7,
    "tokopedia": 7,
    "lazada": 7,
    "商汤": 7,
    "sensetime": 7,
    "旷视": 7,
    "megvii": 7,
    "依图": 7,
    "yitu": 7,
    "云从": 7,
    "cloudwalk": 7,
    "大疆": 7,
    "dji": 7,
    "科大讯飞": 7,
    "iflytek": 7,
    "寒武纪": 7,
    "cambricon": 7,
    "bilibili": 7,
    "哔哩哔哩": 7,
    "小红书": 7,
    "xiaohongshu": 7,
    "知乎": 6,
    "zhihu": 6,
    "携程": 6,
    "ctrip": 6,
    "trip.com": 6,
    "去哪儿": 6,
    "qunar": 6,
    "搜狐": 6,
    "sohu": 6,
    "新浪": 6,
    "sina": 6,
    "微博": 6,
    "weibo": 6,
    "360": 6,
    "奇虎": 6,
    "陌陌": 6,
    "momo": 6,
    "猿辅导": 6,
    "作业帮": 6,
    "好未来": 6,
    "新东方": 6,
    # Tier 3 — other recognizable companies
    "联想": 5,
    "lenovo": 5,
    "dell": 5,
    "hp": 5,
    "ibm": 5,
    "oracle": 5,
    "sap": 5,
    "cisco": 5,
    "intel": 5,
    "amd": 5,
    "qualcomm": 5,
    "samsung": 5,
    "三星": 5,
    "lg": 5,
    "sony": 5,
    "索尼": 5,
    "ebay": 5,
    "paypal": 5,
    "uber": 5,
    "airbnb": 5,
    "twitter": 5,
    "x": 5,
    "linkedin": 5,
    "dropbox": 5,
    "stripe": 5,
    "coinbase": 5,
    "databricks": 5,
    "snowflake": 5,
    "confluent": 5,
    "mongodb": 5,
    "elastic": 5,
    "redis": 5,
    "hashicorp": 5,
    "gitlab": 5,
    "github": 5,
    "atlassian": 5,
    "canva": 5,
    # Chinese financial / SOE tech
    "中金": 6,
    "中信": 6,
    "招商银行": 6,
    "工商银行": 5,
    "建设银行": 5,
    "农业银行": 5,
    "中国银行": 5,
    "平安": 6,
    "中国移动": 5,
    "中国电信": 5,
    "中国联通": 5,
    # Research institutions
    "中科院": 7,
    "中国科学院": 7,
    "微软亚洲研究院": 9,
    "msra": 9,
    "deepmind": 9,
    "google research": 9,
    "fair": 8,
    "ibm research": 7,
}


# ---------------------------------------------------------------------------
# Specificity markers — signals that a description is concrete, not generic
# ---------------------------------------------------------------------------

# Chinese/English number/percentage patterns
_METRIC_PATTERNS = [
    re.compile(p)
    for p in [
        r"\d+[\.\d]*%",           # 30%, 3.5%
        r"\d+[万亿千百]+",         # 10万, 3000亿
        r"\d+[kKmMbB]\+?",        # 10k, 5M, 100K+
        r"\d+\+?\s*(用户|人|次|条|篇|个|台|小时|天|周|月)",  # 100万用户, 50+人
        r"\d+[\.\d]*\s*[qQ][pP][sS]",  # 10 QPS
        r"qps\s*\d+",
        r"[pP]?\d+\s*[mM][sS]",   # 50ms, p99
        r"\d+\s*[gG][bB]",        # 10GB
        r"提升了?\s*\d+",          # 提升了30%
        r"降低了?\s*\d+",
        r"增长了?\s*\d+",
        r"减少了?\s*\d+",
        r"节省了?\s*\d+",
        r"优化了?\s*\d+",
        r"日?[活月][跃均]\s*\d+",   # 日活100万
        r"[dD][aA][uU]\s*\d+",
        r"[mM][aA][uU]\s*\d+",
        r"排名\s*(前|第|top|TOP)\s*\d+",
        r"top\s*\d+",
        r"覆盖\s*\d+",
        r"规模\s*\d+",
        r"\d+\s*行",              # 1000行代码
        r"\d+\s*个?\s*(接口|服务|模块|组件|微服务)",
        r"\d+\s*[a-zA-Z]+",       # 10w+ uv (generic number+unit)
    ]
]

# Action-result Chinese patterns — "通过 X 实现了 Y"
_ACTION_RESULT_PATTERNS = [
    re.compile(p)
    for p in [
        r"通过.{2,20}(提升|降低|减少|增加|实现|完成|达到|节省)",
        r"(负责|主导|参与|独立).{0,30}(开发|设计|搭建|优化|重构|建设|落地)",
        r"基于.{0,20}(搭建|实现|完成|构建|开发)",
        r"(从|由)\d+\s*(到|至|提升到|增长到)\s*\d+",
        r"(改进|改良|重构|重写).{0,30}(系统|模块|服务|架构)",
        r"(设计|实现|开发)了?.{0,30}(系统|平台|工具|框架|引擎|中台)",
        r"(落地|推动|推进).{0,20}(方案|项目|需求|架构)",
        r"(撰写|输出|沉淀).{0,20}(文档|方案|专利|论文)",
    ]
]


def _company_score(company_name: Optional[str]) -> float:
    """Score a single company name against the recognized list."""
    if not company_name:
        return 0.0
    name = str(company_name).strip().lower()
    if not name:
        return 0.0
    # Exact match
    if name in _RECOGNIZED_COMPANIES:
        return float(_RECOGNIZED_COMPANIES[name])
    # Substring match (e.g., "腾讯科技" contains "腾讯")
    for key, score in _RECOGNIZED_COMPANIES.items():
        if key in name or name in key:
            return float(score)
    # Clamp: unrecognized companies get 2.0 — having any company is slightly positive
    return 2.0


def _best_company_score(work_experiences: List[Dict[str, Any]]) -> float:
    """Best company score across all work experiences, scaled to 0-10."""
    best = 0.0
    for exp in work_experiences:
        if not isinstance(exp, dict):
            continue
        company = exp.get("company")
        s = _company_score(company)
        if s > best:
            best = s
    return best


def _description_specificity_score(description: Optional[str]) -> Dict[str, Any]:
    """Evaluate a single description for specificity signals."""
    if not description:
        return {"length": 0, "has_metrics": False, "metric_count": 0,
                "has_action_result": False, "action_result_count": 0}

    text = str(description).strip()
    length = len(text)

    metric_count = sum(1 for p in _METRIC_PATTERNS if p.search(text))
    action_count = sum(1 for p in _ACTION_RESULT_PATTERNS if p.search(text))

    return {
        "length": length,
        "has_metrics": metric_count > 0,
        "metric_count": metric_count,
        "has_action_result": action_count > 0,
        "action_result_count": action_count,
    }


def _all_descriptions(work_experiences: List[Dict[str, Any]],
                      projects: List[Dict[str, Any]]) -> List[str]:
    """Extract all description texts from experiences and projects."""
    texts: List[str] = []
    for exp in (work_experiences or []):
        if isinstance(exp, dict) and exp.get("description"):
            texts.append(str(exp["description"]).strip())
    for proj in (projects or []):
        if isinstance(proj, dict) and proj.get("description"):
            texts.append(str(proj["description"]).strip())
    return texts


def score_candidate_experience_quality(
    candidate: CandidateModel,
) -> Dict[str, Any]:
    """Rule-based experience quality scoring (0-100).

    Evaluates what HR can see without technical understanding:
    1. Content fullness — are descriptions substantial or empty?
    2. Specificity — are there concrete numbers, tools, outcomes?
    3. Company prestige — any recognizable company names?
    4. Experience breadth — how many entries?

    Returns same structure as the LLM version for compatibility.
    """
    work_exps = candidate.work_experience or []
    projects = candidate.projects or []
    skills = candidate.skills or []

    descriptions = _all_descriptions(work_exps, projects)
    spec_results = [_description_specificity_score(d) for d in descriptions]

    # ------------------------------------------------------------------
    # Dimension 1: Content fullness (0-100)
    # ------------------------------------------------------------------
    # HR instantly notices empty vs substantial descriptions
    total_desc_count = len(descriptions)
    if total_desc_count == 0:
        content_score = 0.0
    else:
        # Score each description by length (钟形曲线):
        # < 20 chars  → 0   (empty / placeholder)
        # 20-50       → 15  (bare minimum)
        # 50-100      → 30  (a short paragraph)
        # 100-200     → 45  (接近及格但不够)
        # 200-250     → 60  (及格)
        # 250-350     → 78  (充实)
        # 350-450     → 100 (很详细 — 最佳印象)
        # 450+        → 70  (过长扣分 — HR找不到重点)
        desc_scores = []
        for r in spec_results:
            length = r["length"]
            if length < 20:
                desc_scores.append(0.0)
            elif length < 50:
                desc_scores.append(15.0)
            elif length < 100:
                desc_scores.append(30.0)
            elif length < 200:
                desc_scores.append(45.0)
            elif length < 250:
                desc_scores.append(60.0)
            elif length < 350:
                desc_scores.append(78.0)
            elif length < 450:
                desc_scores.append(100.0)
            else:
                desc_scores.append(70.0)

        # Weight: best descriptions matter more than worst
        # Take mean but give extra weight to the best one
        content_score = sum(desc_scores) / len(desc_scores)

    # ------------------------------------------------------------------
    # Dimension 2: Specificity (0-100)
    # ------------------------------------------------------------------
    # Numbers, action-result patterns, concrete tools
    if total_desc_count == 0:
        specificity_score = 0.0
    else:
        total_metrics = sum(r["metric_count"] for r in spec_results)
        total_actions = sum(r["action_result_count"] for r in spec_results)

        # Per-description averages
        avg_metrics = total_metrics / total_desc_count
        avg_actions = total_actions / total_desc_count

        # Map to 0-100:
        # 0 metrics + 0 actions → 0
        # 1+ metrics or 1+ actions → 50 (some specifics)
        # 3+ combined → 100 (very specific)
        combined = avg_metrics + avg_actions
        if combined >= 3:
            specificity_score = 100.0
        elif combined >= 2:
            specificity_score = 80.0
        elif combined >= 1:
            specificity_score = 50.0
        elif combined >= 0.5:
            specificity_score = 25.0
        else:
            specificity_score = 0.0

        # Bonus: having at least one description with both metrics AND actions
        has_both = any(r["has_metrics"] and r["has_action_result"] for r in spec_results)
        if has_both:
            specificity_score = min(100.0, specificity_score + 15.0)

    # ------------------------------------------------------------------
    # Dimension 3: Company prestige (0-100)
    # ------------------------------------------------------------------
    best_company = _best_company_score(work_exps)
    # Map company tier to 0-100:
    # Tier 1 (10)  → 100
    # Tier 2 (7-9) → 70-90
    # Tier 3 (5-6) → 40-60
    # Unknown (2)  → 15
    # None (0)     → 0
    company_score = best_company * 10.0  # 0-10 → 0-100 linear

    # ------------------------------------------------------------------
    # Dimension 4: Experience breadth (0-100)
    # ------------------------------------------------------------------
    work_count = len([e for e in work_exps if isinstance(e, dict)
                      and str(e.get("description") or "").strip()])
    proj_count = len([p for p in projects if isinstance(p, dict)
                      and str(p.get("description") or "").strip()])
    has_skills = len(skills) > 0
    has_summary = bool(candidate.summary and str(candidate.summary).strip())

    # Breadth points:
    breadth_points = 0.0
    breadth_points += min(work_count, 5) * 10.0  # max 50 for work
    breadth_points += min(proj_count, 3) * 10.0  # max 30 for projects
    if has_skills:
        breadth_points += 10.0  # skills listed
    if has_summary:
        breadth_points += 10.0  # personal summary
    breadth_score = min(100.0, breadth_points)

    # ------------------------------------------------------------------
    # Combine into overall quality score
    # ------------------------------------------------------------------
    # Weights: content 35%, specificity 30%, company 20%, breadth 15%
    quality_score = round(
        0.35 * content_score
        + 0.30 * specificity_score
        + 0.20 * company_score
        + 0.15 * breadth_score,
        1,
    )

    # Map component scores to LLM-compatible field names for drop-in compatibility
    impact_score = round(
        0.4 * company_score + 0.3 * breadth_score + 0.3 * content_score * 0.5,
        1,
    )
    evidence_quality_score = round(
        0.5 * content_score + 0.5 * specificity_score,
        1,
    )
    consistency_risk = round(
        max(0.0, 100.0 - (0.6 * content_score + 0.4 * specificity_score)),
        1,
    )

    # Build a human-readable summary
    summary_parts = []
    if best_company >= 8:
        summary_parts.append("有大厂实习经历")
    elif best_company >= 5:
        summary_parts.append("有知名公司经历")
    if total_desc_count >= 3:
        summary_parts.append(f"共{total_desc_count}段经历/项目描述")
    elif total_desc_count == 0:
        summary_parts.append("缺少实习和项目描述")
    else:
        summary_parts.append(f"有{total_desc_count}段经历/项目描述")
    if content_score >= 70:
        summary_parts.append("描述较为充实")
    elif content_score < 30 and total_desc_count > 0:
        summary_parts.append("经历描述偏简短，缺少细节")
    if specificity_score >= 50:
        summary_parts.append("描述中包含具体数据和成果")
    elif total_desc_count > 0:
        summary_parts.append("描述中缺少量化数据和具体成果")
    if not skills:
        summary_parts.append("未填写技能标签")
    if not work_exps and not projects:
        summary_parts.append("无实习和工作经历")

    summary_text = "；".join(summary_parts) if summary_parts else "无经历数据"

    return {
        "impact_score": impact_score,
        "evidence_quality_score": evidence_quality_score,
        "consistency_risk": consistency_risk,
        "llm_quality_score": quality_score,
        "summary": summary_text,
        "status": "rule_based",
    }
