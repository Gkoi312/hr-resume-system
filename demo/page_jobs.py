"""
Job Management page logic — imported by app.py.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client as api


def render():
    st.title("📌 岗位管理")
    st.caption("管理 JD（Job Description）：创建、上传文件解析、查看结构化结果")

    if not api.require_login():
        return

    tab_list, tab_create, tab_upload = st.tabs(["📋 岗位列表", "✏️ 手动创建", "📤 上传 JD（岗位描述）"])

    with tab_list:
        _render_list()
    with tab_create:
        _render_create()
    with tab_upload:
        _render_upload()


def _structured_card(structured: dict) -> None:
    if not structured:
        st.caption("暂无结构化数据")
        return
    cols = st.columns([1, 1])
    with cols[0]:
        job_title = structured.get("job_title")
        if job_title:
            st.markdown(f"**职位名称**: {job_title}")
        req_skills = structured.get("required_skills", [])
        if req_skills:
            st.markdown("**必备技能**: " + " | ".join(req_skills))
        pref_skills = structured.get("preferred_skills", [])
        if pref_skills:
            st.markdown("**优先技能**: " + " | ".join(pref_skills))
    with cols[1]:
        edu = structured.get("education_requirement")
        if edu:
            st.markdown(f"**学历要求**: {edu}")
        min_years = structured.get("min_years")
        if min_years is not None:
            st.markdown(f"**最低年限**: {min_years} 年")
        industry = structured.get("industry_preference", [])
        if industry:
            st.markdown(f"**行业偏好**: {', '.join(industry)}")
    resp = structured.get("responsibilities", [])
    if resp:
        st.markdown("**岗位职责**")
        for i, r in enumerate(resp):
            st.markdown(f"{i+1}. {r}")
    summary = structured.get("job_summary")
    if summary:
        st.markdown(f"**摘要**: {summary}")


def _render_list():
    try:
        jobs = api.list_jobs(limit=200)
    except Exception as e:
        st.error(f"加载岗位列表失败：{e}")
        return
    if not jobs:
        st.info("暂无岗位，请先创建", icon="ℹ️")
        return
    st.caption(f"共 {len(jobs)} 个岗位")
    for job in jobs:
        with st.container(border=True):
            jc1, jc2 = st.columns([4, 1])
            with jc1:
                st.markdown(f"**{job.get('title', '?')}**")
            with jc2:
                status = job.get("status", "?")
                color = "🟢" if status == "active" else "⚪"
                st.caption(f"{color} {status}")
            with st.expander("📋 查看详情"):
                t1, t2, t3 = st.tabs(["原始文本", "结构化", "JSON"])
                with t1:
                    raw = job.get("raw_jd_text", "")
                    if raw:
                        st.text_area("raw_jd_text", raw, height=200, label_visibility="collapsed")
                    else:
                        st.caption("无原始 JD 文本")
                with t2:
                    _structured_card(job.get("structured") or {})
                with t3:
                    st.json(job)


def _render_create():
    st.subheader("手动创建岗位")
    with st.form("job_create_form"):
        title = st.text_input("岗位名称 *", placeholder="如：后端开发实习生")
        raw_text = st.text_area("JD 原始文本", height=150, placeholder="粘贴 JD 全文…")
        col1, col2 = st.columns(2)
        with col1:
            req_skills = st.text_input("必备技能（逗号分隔）", placeholder="Python, FastAPI, PostgreSQL")
        with col2:
            pref_skills = st.text_input("优先技能（逗号分隔）", placeholder="Docker, Redis")
        col3, col4, col5 = st.columns(3)
        with col3:
            edu_req = st.selectbox("学历要求", ["", "大专", "本科", "硕士", "博士"])
        with col4:
            min_years = st.number_input("最低年限", min_value=0, max_value=20, value=0)
        with col5:
            industry = st.text_input("行业偏好（逗号分隔）", placeholder="互联网, 金融")
        resp_text = st.text_area("岗位职责（每行一条）", height=100, placeholder="负责后端 API 开发\n参与系统架构设计")

        if st.form_submit_button("✅ 创建岗位", use_container_width=True):
            if not title.strip():
                st.error("岗位名称不能为空")
                return
            structured = {}
            if req_skills.strip():
                structured["required_skills"] = [s.strip() for s in req_skills.split(",") if s.strip()]
            if pref_skills.strip():
                structured["preferred_skills"] = [s.strip() for s in pref_skills.split(",") if s.strip()]
            if edu_req:
                structured["education_requirement"] = edu_req
            if min_years > 0:
                structured["min_years"] = min_years
            if industry.strip():
                structured["industry_preference"] = [s.strip() for s in industry.split(",") if s.strip()]
            if resp_text.strip():
                structured["responsibilities"] = [r.strip() for r in resp_text.split("\n") if r.strip()]
            if title.strip():
                structured["job_title"] = title.strip()
            try:
                api.create_job(title=title.strip(), raw_jd_text=raw_text, structured=structured)
                st.success("✅ 创建成功！")
                st.rerun()
            except Exception as e:
                st.error(str(e))


def _render_upload():
    st.subheader("上传 JD 文件（岗位描述）并自动解析")
    st.caption("支持 PDF / DOCX / TXT 格式。简历上传请切换到「简历 & 候选人」页面。")
    uploaded_file = st.file_uploader("选择 JD 文件", type=["pdf", "docx", "txt"], key="jd_uploader")
    if uploaded_file:
        st.markdown(f"📎 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        if st.button("🚀 上传并解析", use_container_width=True, type="primary"):
            with st.spinner("正在解析 JD 文件…"):
                try:
                    job = api.upload_job(file_bytes=uploaded_file.read(), file_name=uploaded_file.name)
                    st.success(f"✅ 解析成功！岗位: **{job.get('title', '?')}**")
                    st.json(job.get("structured", {}))
                    st.rerun()
                except Exception as e:
                    st.error(f"解析失败：{e}")
