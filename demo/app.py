"""
HR Resume Screening System — Demo Frontend

Streamlit-based demo that consumes the FastAPI backend.
Run:  cd demo && streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be the first st. call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Resume Screening",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Shared sidebar: Backend + Auth + Navigation
# ---------------------------------------------------------------------------
import api_client as api

with st.sidebar:
    # ── Navigation ──
    st.markdown("## 📋 功能导航")

    PAGES = {
        "🏠 首页": "home",
        "📌 岗位管理": "jobs",
        "📄 简历 & 候选人": "resumes",
        "🎯 匹配分析": "matching",
    }

    current_page = st.radio(
        "选择页面",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown("---")

    # ── Backend ──
    st.markdown("## ⚙️ Backend")

    st.text_input(
        "API Base URL",
        value=api.BASE_URL_DEFAULT,
        key="hr_api_base_url",
    )

    if st.button("🔍 检测连接"):
        ok = api.health_check()
        if ok:
            st.success("✅ 后端连接正常")
        else:
            st.error("❌ 无法连接后端")

    st.markdown("---")

    # ── Auth ──
    st.markdown("## 👤 账户")

    if api.is_logged_in():
        user = st.session_state.get("hr_api_user", {})
        st.success(f"已登录：**{user.get('username', '?')}**")
        if st.button("🚪 退出登录"):
            api.logout()
            st.rerun()
    else:
        auth_tab1, auth_tab2 = st.tabs(["登录", "注册"])
        with auth_tab1:
            st.text_input("用户名", key="login_user")
            st.text_input("密码", type="password", key="login_pass")
            if st.button("登录", use_container_width=True):
                try:
                    api.login(st.session_state.login_user, st.session_state.login_pass)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with auth_tab2:
            st.text_input("用户名", key="reg_user")
            st.text_input("密码", type="password", key="reg_pass")
            st.text_input("职位", value="HR", key="reg_occ")
            if st.button("注册", use_container_width=True):
                try:
                    api.register(st.session_state.reg_user, st.session_state.reg_pass, st.session_state.reg_occ)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.markdown("---")

# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------
page_id = PAGES[current_page]

if page_id == "home":
    st.title("📋 HR 简历智能筛选系统")
    st.caption("面向校招/实习场景的简历自动筛选与匹配分析平台")

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("### 📌 岗位管理")
            st.markdown("创建岗位、上传 JD 文件自动结构化解析")
    with col2:
        with st.container(border=True):
            st.markdown("### 📄 简历 & 候选人")
            st.markdown("批量上传简历，自动构建候选人画像")
    with col3:
        with st.container(border=True):
            st.markdown("### 🎯 匹配分析")
            st.markdown("学历过滤 + 多轴语义打分，可解释排序")

    if api.is_logged_in():
        st.success("✅ 已登录，在左侧边栏点击功能名称开始使用。")
    else:
        st.warning("⚠️ 请先在左侧边栏登录/注册")

elif page_id == "jobs":
    from page_jobs import render as render_jobs
    render_jobs()

elif page_id == "resumes":
    from page_resumes import render as render_resumes
    render_resumes()

elif page_id == "matching":
    from page_matching import render as render_matching
    render_matching()
