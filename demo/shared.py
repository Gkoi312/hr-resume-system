"""
Shared sidebar and helpers used by app.py and all pages/.
"""
from __future__ import annotations

import streamlit as st
import api_client as api


def render_sidebar():
    """Render backend config + auth in the sidebar. Call once per page."""
    with st.sidebar:
        st.markdown("## ⚙️ Backend")

        st.text_input(
            "API Base URL",
            value=api.BASE_URL_DEFAULT,
            key="hr_api_base_url",
            help="FastAPI 后端的 base URL",
        )

        if st.button("🔍 检测连接"):
            ok = api.health_check()
            if ok:
                st.success("✅ 后端连接正常")
            else:
                st.error("❌ 无法连接后端")

        st.markdown("---")
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
                        api.login(
                            st.session_state.login_user,
                            st.session_state.login_pass,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            with auth_tab2:
                st.text_input("用户名", key="reg_user")
                st.text_input("密码", type="password", key="reg_pass")
                st.text_input("职位", value="HR", key="reg_occ")
                if st.button("注册", use_container_width=True):
                    try:
                        api.register(
                            st.session_state.reg_user,
                            st.session_state.reg_pass,
                            st.session_state.reg_occ,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.markdown("---")
