"""
Reusable match card & explanation rendering components.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def score_bar(label: str, score: Optional[float], max_val: float = 1.0, help_text: str = "") -> None:
    """Render a labelled progress bar for a single score."""
    if score is None:
        st.metric(label, "N/A")
        return
    pct = min(float(score) / max_val, 1.0)
    color = (
        "🟢" if pct >= 0.7 else
        "🟡" if pct >= 0.4 else
        "🔴"
    )
    st.markdown(f"**{label}**: {color} {score:.3f}" + (f"  _{help_text}_" if help_text else ""))


def render_score_breakdown(breakdown: Optional[Dict[str, Any]]) -> None:
    """Render ScoreBreakdown as metrics row."""
    if not breakdown:
        st.caption("暂无分数详情")
        return
    cols = st.columns(4)
    with cols[0]:
        st.metric("综合分", f"{breakdown.get('overall_score', 0):.3f}" if breakdown.get("overall_score") is not None else "N/A")
    with cols[1]:
        sk = breakdown.get("skill_score")
        sw = breakdown.get("skill_weight")
        st.metric("技能分", f"{sk:.3f}" if sk is not None else "N/A", delta=f"权重 {sw}" if sw else None)
    with cols[2]:
        se = breakdown.get("semantic_score")
        sew = breakdown.get("semantic_weight")
        st.metric("语义分", f"{se:.3f}" if se is not None else "N/A", delta=f"权重 {sew}" if sew else None)
    with cols[3]:
        ll = breakdown.get("llm_quality_score")
        lw = breakdown.get("llm_quality_weight")
        st.metric("LLM质量分", f"{ll:.3f}" if ll is not None else "N/A", delta=f"权重 {lw}" if lw else None)


def render_semantic_evidence(evidence: Optional[List[Dict[str, Any]]]) -> None:
    """Render semantic evidence snippets."""
    if not evidence:
        st.caption("暂无语义证据")
        return
    for i, snippet in enumerate(evidence):
        src = snippet.get("source_type", "?")
        text = snippet.get("text", "")
        score = snippet.get("score")
        rationale = snippet.get("rationale")
        with st.container(border=True):
            st.caption(f"📎 证据 #{i+1} · 来源: **{src}**" + (f" · 分数: {score:.3f}" if score is not None else ""))
            st.markdown(text[:600] + ("…" if len(text) > 600 else ""))
            if rationale:
                st.caption(f"💡 {rationale}")


def render_delivery_alignments(alignments: Optional[List[Dict[str, Any]]]) -> None:
    """Render delivery (职责↔经历) alignment items as a detailed table."""
    if not alignments:
        st.caption("暂无 Delivery 对齐数据")
        return

    rows = []
    for a in alignments:
        rows.append({
            "岗位职责": a.get("job_text_snippet", "")[:150],
            "候选人经历": a.get("cand_text_snippet", "")[:150],
            "Cosine": f"{a.get('cosine', 0):.3f}",
            "BM25": f"{a.get('bm25', 0):.3f}",
            "RRF": f"{a.get('rrf', 0):.3f}",
            "cos排名": a.get("rank_cos", ""),
            "bm25排名": a.get("rank_bm25", ""),
            "shared_terms": ", ".join(a.get("shared_terms", [])[:10]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_llm_quality(llm: Optional[Dict[str, Any]]) -> None:
    """Render LLM quality assessment details."""
    if not llm:
        st.caption("暂无 LLM 质量评估")
        return
    status = llm.get("status", "?")
    st.markdown(f"**状态**: {status}")
    cols = st.columns(3)
    with cols[0]:
        st.metric("影响力", f"{llm.get('impact_score', 0):.2f}" if llm.get("impact_score") is not None else "N/A")
    with cols[1]:
        st.metric("证据质量", f"{llm.get('evidence_quality_score', 0):.2f}" if llm.get("evidence_quality_score") is not None else "N/A")
    with cols[2]:
        st.metric("一致性风险", f"{llm.get('consistency_risk', 0):.2f}" if llm.get('consistency_risk') is not None else "N/A")
    summary = llm.get("summary")
    if summary:
        st.caption(f"📝 {summary}")


def render_match_card(match: Dict[str, Any], rank: int = 0) -> None:
    """
    Render a single match result as an expandable card.

    Args:
        match: MatchWithCandidate dict from GET /matching/job/{id}
        rank: 1-based rank for display
    """
    score = match.get("overall_score")
    score_str = f"{score:.3f}" if score is not None else "?"
    rec = match.get("recommendation", "")
    action = ""
    explanation = match.get("explanation") or {}

    # Suggested action coloring
    suggested = (explanation or {}).get("suggested_action", "")
    action_icon = {
        "recommend_interview": "🟢 建议面试",
        "further_screening": "🟡 待进一步筛选",
        "not_recommended": "🔴 不推荐",
    }.get(suggested, "")

    with st.container(border=True):
        # Header row
        hdr_cols = st.columns([0.5, 2, 1, 1, 1])
        with hdr_cols[0]:
            st.markdown(f"### #{rank}")
        with hdr_cols[1]:
            name = match.get("candidate_name") or explanation.get("candidate_name", "?")
            email = match.get("candidate_email") or ""
            st.markdown(f"**{name}**  {f'({email})' if email else ''}")
            if action_icon:
                st.caption(action_icon)
        with hdr_cols[2]:
            st.metric("综合分", score_str)
        with hdr_cols[3]:
            sem = match.get("semantic_score")
            st.metric("语义分", f"{sem:.3f}" if sem is not None else "?")
        with hdr_cols[4]:
            sk = match.get("skill_score")
            st.metric("技能分", f"{sk:.3f}" if sk is not None else "?")

        # Expandable detail
        with st.expander("📋 查看详细匹配报告", expanded=(rank <= 3)):
            render_explanation_detail(explanation)


def render_explanation_detail(explanation: Dict[str, Any]) -> None:
    """Render the full MatchExplanation in detail."""
    # Hard requirements & signals
    col_left, col_right = st.columns(2)
    with col_left:
        hard = explanation.get("hard_requirements_met", [])
        missing = explanation.get("missing_requirements", [])
        if hard:
            st.success(f"✅ 满足硬门槛: {', '.join(hard)}")
        if missing:
            st.error(f"❌ 缺失硬门槛: {', '.join(missing)}")

        strong = explanation.get("strong_signals", [])
        if strong:
            st.markdown("**🌟 亮点信号**")
            for s in strong:
                st.markdown(f"- {s}")

    with col_right:
        risk = explanation.get("risk_signals", [])
        if risk:
            st.markdown("**⚠️ 风险信号**")
            for r in risk:
                st.markdown(f"- {r}")

    # HR Summary
    summary = explanation.get("summary_for_hr")
    if summary:
        st.markdown("---")
        st.markdown(f"**📝 HR 摘要**: {summary}")

    # Interview focus
    focus = explanation.get("interview_focus_points", [])
    if focus:
        st.markdown("**🎙️ 面试建议追问**")
        for f in focus:
            st.markdown(f"- {f}")

    # Score breakdown
    st.markdown("---")
    st.markdown("#### 📊 分数拆解")
    render_score_breakdown(explanation.get("score_breakdown"))

    # Skills comparison
    st.markdown("---")
    sk_col1, sk_col2 = st.columns(2)
    with sk_col1:
        matched = explanation.get("matched_skills", [])
        st.markdown(f"**✅ 匹配技能** ({len(matched)})")
        if matched:
            st.markdown(" | ".join(matched))
    with sk_col2:
        missing_s = explanation.get("missing_skills", [])
        st.markdown(f"**❌ 缺失技能** ({len(missing_s)})")
        if missing_s:
            st.markdown(" | ".join(missing_s))

    # Semantic evidence
    st.markdown("---")
    st.markdown("#### 🔍 语义证据")
    render_semantic_evidence(explanation.get("semantic_evidence"))

    # Delivery alignments
    st.markdown("---")
    st.markdown("#### 🔗 职责↔经历 对齐 (Delivery Alignment)")
    render_delivery_alignments(explanation.get("delivery_alignments"))

    # LLM quality
    st.markdown("---")
    st.markdown("#### 🤖 LLM 质量评估")
    render_llm_quality(explanation.get("llm_quality"))
