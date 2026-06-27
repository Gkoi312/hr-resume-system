"""
Resume & Candidate management page logic — imported by app.py.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client as api


def render():
    st.title("📄 简历 & 候选人")
    st.caption("上传简历 → 自动解析 → 构建候选人画像")

    if not api.require_login():
        return

    tab_upload, tab_candidates = st.tabs(["📤 上传简历（支持批量）", "👥 候选人库"])

    with tab_upload:
        _render_upload()
    with tab_candidates:
        _render_candidates()


def _render_candidate_profile(candidate: dict) -> None:
    st.markdown("#### 🎓 教育经历")
    edu_list = candidate.get("education") or []
    if edu_list:
        for edu in edu_list:
            if isinstance(edu, dict):
                school = edu.get("school", edu.get("institution", "?"))
                degree = edu.get("degree", "")
                major = edu.get("major", "")
                st.markdown(f"- **{school}** · {degree} · {major}")
            else:
                st.markdown(f"- {edu}")
    else:
        st.caption("无")

    st.markdown("#### 💼 工作/实习经历")
    work_list = candidate.get("work_experience") or []
    if work_list:
        for w in work_list:
            if isinstance(w, dict):
                company = w.get("company", "?")
                title = w.get("title", w.get("position", ""))
                duration = w.get("duration", "")
                desc = w.get("description", w.get("summary", ""))
                st.markdown(f"- **{company}** · {title}  _({duration})_")
                if desc:
                    st.caption(f"  {str(desc)[:200]}")
            else:
                st.markdown(f"- {w}")
    else:
        st.caption("无")

    st.markdown("#### 🚀 项目经历")
    proj_list = candidate.get("projects") or []
    if proj_list:
        for p in proj_list:
            if isinstance(p, dict):
                name = p.get("name", "?")
                desc = p.get("description", "")
                st.markdown(f"- **{name}**")
                if desc:
                    st.caption(f"  {str(desc)[:200]}")
            else:
                st.markdown(f"- {p}")
    else:
        st.caption("无")

    st.markdown("#### 🏷️ 技能标签")
    skills = candidate.get("skills") or []
    if skills:
        st.markdown(" | ".join(skills))
    else:
        st.caption("无")

    tags = candidate.get("direction_tags") or []
    if tags:
        st.markdown("**方向标签**: " + ", ".join(tags))


def _render_upload():
    st.subheader("批量上传简历")
    st.caption("支持 PDF / DOCX / TXT / PNG / JPG，可一次选择多个文件。每份简历自动创建候选人画像。")

    uploaded_files = st.file_uploader(
        "选择一个或多个简历文件",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="resume_uploader",
    )

    if uploaded_files:
        n = len(uploaded_files)
        st.markdown(f"已选择 **{n}** 个文件：")
        for f in uploaded_files:
            st.caption(f"  📎 {f.name} ({f.size / 1024:.1f} KB)")

        if st.button(f"🚀 上传并解析（{n} 份）", use_container_width=True, type="primary"):
            ok, fail = [], []
            bar = st.progress(0, f"0/{n}")
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.markdown(f"⏳ 正在解析 **{file.name}** …")
                try:
                    result = api.upload_resume(file_bytes=file.read(), file_name=file.name)
                    ok.append({
                        "文件": file.name,
                        "大小(KB)": f"{file.size / 1024:.1f}",
                        "Resume ID": result.get("id", "?")[:8] + "…",
                        "Candidate ID": result.get("candidate_id", "?")[:8] + "…",
                        "状态": "✅",
                    })
                except Exception as e:
                    fail.append({"文件": file.name, "大小(KB)": f"{file.size / 1024:.1f}", "错误": str(e)[:200], "状态": "❌"})
                bar.progress((i + 1) / n, f"{i + 1}/{n}")

            bar.empty()
            status_text.empty()
            st.success(f"🎉 完成！成功 {len(ok)} / 失败 {len(fail)}")
            if ok:
                st.dataframe(pd.DataFrame(ok), use_container_width=True, hide_index=True)
            if fail:
                for f in fail:
                    st.error(f"{f['文件']}: {f['错误']}")
            if ok:
                st.button("🔄 刷新候选人库", on_click=st.rerun, use_container_width=True)

    st.markdown("---")
    st.caption("💡 提示：如果后端配置了 LLM 解析，上传速度会更快且更准确。")


def _render_candidates():
    st.subheader("候选人库")
    fc1, fc2, fc3, _ = st.columns([2, 2, 2, 2])
    with fc1:
        keyword = st.text_input("关键词搜索", placeholder="姓名/邮箱")
    with fc2:
        skill_filter = st.text_input("技能过滤", placeholder="Python")
    with fc3:
        edu_filter = st.selectbox("学历过滤", ["", "大专", "本科", "硕士", "博士"])

    try:
        candidates = api.list_candidates(
            keyword=keyword or None,
            skill=skill_filter or None,
            education=edu_filter or None,
            limit=200,
        )
    except Exception as e:
        st.error(f"加载候选人失败：{e}")
        return

    if not candidates:
        st.info("暂无候选人，请先上传简历", icon="ℹ️")
        return

    st.caption(f"共 {len(candidates)} 位候选人")
    for cand in candidates:
        with st.container(border=True):
            cc1, cc2, cc3 = st.columns([2, 1, 1])
            with cc1:
                name = cand.get("name", "?")
                email = cand.get("email", "")
                st.markdown(f"**{name}**  {f'({email})' if email else ''}")
            with cc2:
                years = cand.get("years_of_experience")
                st.caption(f"工作年限: {years if years is not None else '?'}")
            with cc3:
                cid = cand.get("id", "")
                st.caption(f"ID: {cid[:8]}…")
            with st.expander("📋 查看候选人画像"):
                _render_candidate_profile(cand)
                st.markdown("---")
                st.json(cand)
