"""
Matching Analysis page logic — imported by app.py.
"""
from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import streamlit as st

import api_client as api
from components.match_card import render_match_card


def render():
    st.title("🎯 匹配分析")
    st.caption("选择岗位，对候选人池进行智能匹配打分，查看排序结果与可解释证据")

    if not api.require_login():
        return

    # ── Step 1: Select Job ──────────────────────────────────────────
    st.markdown("### 📌 第一步：选择岗位")
    try:
        jobs = api.list_jobs(limit=200)
    except Exception as e:
        st.error(f"加载岗位列表失败：{e}")
        return

    if not jobs:
        st.info("暂无岗位，请先在「岗位管理」页面创建岗位", icon="ℹ️")
        return

    job_options = {f"{j.get('title', '?')} ({j['id'][:8]}…)": j["id"] for j in jobs}
    selected_label = st.selectbox("岗位", list(job_options.keys()), key="match_job_selector")
    selected_job_id = job_options[selected_label]

    job_detail = next((j for j in jobs if j["id"] == selected_job_id), None)
    if job_detail:
        with st.expander("📋 查看岗位详情", expanded=False):
            c1, c2 = st.columns(2)
            structured = job_detail.get("structured") or {}
            with c1:
                req = structured.get("required_skills", [])
                st.markdown(f"**必备技能**: {', '.join(req) if req else '未指定'}")
                pref = structured.get("preferred_skills", [])
                st.markdown(f"**优先技能**: {', '.join(pref) if pref else '未指定'}")
            with c2:
                st.markdown(f"**学历要求**: {structured.get('education_requirement', '未指定')}")
                st.markdown(f"**最低年限**: {structured.get('min_years', '未指定')} 年")
            resp = structured.get("responsibilities", [])
            if resp:
                st.markdown("**职责**:")
                for r in resp:
                    st.markdown(f"- {r}")

    # ── Step 2: Actions ────────────────────────────────────────────
    st.markdown("### ⚡ 第二步：执行操作")
    ac1, ac2, ac3 = st.columns([1, 1, 2])
    with ac1:
        run_btn = st.button("🚀 运行匹配", use_container_width=True, type="primary")
    with ac2:
        edu_btn = st.button("🔍 学历过滤预览", use_container_width=True)
    with ac3:
        load_btn = st.button("📊 加载已有匹配结果", use_container_width=True)

    # ── Education preview ──────────────────────────────────────────
    if edu_btn:
        with st.spinner("正在计算学历过滤…"):
            try:
                preview = api.education_filter(selected_job_id)
                st.markdown("---")
                st.markdown("#### 🔍 学历过滤预览")
                total = preview.get("total_input", 0)
                passed = preview.get("passed_count", 0)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("总候选人数", total)
                mc2.metric("通过学历过滤", passed)
                mc3.metric("未通过", total - passed)
                candidates = preview.get("candidates", [])
                if candidates:
                    df = pd.DataFrame([{
                        "候选人": c.get("candidate_name", "?"),
                        "学历等级": c.get("resume_best_degree_rank", "?"),
                        "学历来源": ", ".join(c.get("resume_degree_levels", [])),
                        "要求等级": c.get("education_required_min_rank", "?"),
                        "通过": "✅" if c.get("meets_requirement") else "❌",
                    } for c in candidates])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"学历过滤失败：{e}")

    # ── Run matching ───────────────────────────────────────────────
    if run_btn:
        st.markdown("---")
        st.markdown("#### ⚡ 匹配进度")
        bar = st.progress(0, "正在运行匹配…")
        t0 = time.time()
        try:
            results = api.run_matching(selected_job_id)
            elapsed = time.time() - t0
            bar.progress(1.0, f"✅ 完成！耗时 {elapsed:.1f}s")
        except Exception as e:
            bar.empty()
            st.error(f"匹配失败：{e}")
            return
        if not results:
            st.warning("⚠️ 没有产生匹配结果。可能所有候选人均未通过学历过滤，或候选人库为空。")
        else:
            st.success(f"🎉 匹配完成！共 {len(results)} 位候选人通过学历过滤并获得打分")
            st.session_state["match_results"] = results
            st.session_state["match_job_id"] = selected_job_id

    # ── Load existing ──────────────────────────────────────────────
    if load_btn:
        with st.spinner("加载已有匹配结果…"):
            try:
                results = api.get_matches_by_job(selected_job_id, limit=200)
                if not results:
                    st.warning("该岗位暂无匹配记录，请先运行匹配")
                else:
                    st.success(f"已加载 {len(results)} 条匹配记录")
                    st.session_state["match_results"] = results
                    st.session_state["match_job_id"] = selected_job_id
            except Exception as e:
                st.error(f"加载失败：{e}")

    # ── Step 3: Results ────────────────────────────────────────────
    results: Optional[List[dict]] = st.session_state.get("match_results")
    if results is not None and st.session_state.get("match_job_id") == selected_job_id:
        st.markdown("---")
        st.markdown("### 📊 匹配结果")

        total = len(results)
        avg_score = sum(r.get("overall_score") or 0 for r in results) / total if total else 0
        recommended = sum(1 for r in results if (r.get("explanation") or {}).get("suggested_action") == "recommend_interview")
        screening = sum(1 for r in results if (r.get("explanation") or {}).get("suggested_action") == "further_screening")

        for col, label, val in zip(
            st.columns(5),
            ["匹配人数", "平均分", "建议面试", "待筛选", "最高分"],
            [total, f"{avg_score:.3f}", recommended, screening, f"{max(r.get('overall_score') or 0 for r in results):.3f}"],
        ):
            col.metric(label, val)

        st.markdown("---")
        for rank, match in enumerate(results, start=1):
            render_match_card(match, rank=rank)

        with st.expander("🔧 原始 JSON 数据"):
            st.json(results)
