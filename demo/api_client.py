"""
API client for the HR Resume System FastAPI backend.

Thin wrapper around requests that handles JWT auth, base URL config,
and typed response helpers. All functions return parsed JSON (dict/list).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

BASE_URL_DEFAULT = os.environ.get("HR_API_BASE_URL", "http://127.0.0.1:8000/api/v1")

SESSION_KEYS = {
    "base_url": "hr_api_base_url",
    "token": "hr_api_token",
    "user": "hr_api_user",
}


def _base() -> str:
    return st.session_state.get(SESSION_KEYS["base_url"], BASE_URL_DEFAULT)


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    token = st.session_state.get(SESSION_KEYS["token"])
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _ok(r: requests.Response) -> requests.Response:
    """Raise a Streamlit-friendly error on non-2xx."""
    if not r.ok:
        detail = ""
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:500]
        # 401 means token expired / invalid → clear session
        if r.status_code == 401:
            st.session_state.pop(SESSION_KEYS["token"], None)
            st.session_state.pop(SESSION_KEYS["user"], None)
        raise RuntimeError(f"[{r.status_code}] {detail}")
    return r


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login(username: str, password: str) -> dict:
    r = _ok(requests.post(
        f"{_base()}/auth/login",
        json={"username": username, "password": password},
        headers={"Content-Type": "application/json"},
    ))
    data = r.json()
    st.session_state[SESSION_KEYS["token"]] = data["access_token"]
    st.session_state[SESSION_KEYS["user"]] = data.get("user", {})
    return data


def register(username: str, password: str, occupation: str = "HR") -> dict:
    r = _ok(requests.post(
        f"{_base()}/auth/register",
        json={"username": username, "password": password, "occupation": occupation},
        headers={"Content-Type": "application/json"},
    ))
    data = r.json()
    st.session_state[SESSION_KEYS["token"]] = data["access_token"]
    st.session_state[SESSION_KEYS["user"]] = data.get("user", {})
    return data


def is_logged_in() -> bool:
    return SESSION_KEYS["token"] in st.session_state


def require_login() -> bool:
    """Check login; if not, show a warning and return False. Call at page top."""
    if not is_logged_in():
        st.warning("⚠️ 请先在左侧边栏登录后再使用此功能")
        return False
    # Show current user
    user = st.session_state.get(SESSION_KEYS["user"], {})
    st.caption(f"👤 当前用户：**{user.get('username', '?')}**")
    return True


def logout() -> None:
    st.session_state.pop(SESSION_KEYS["token"], None)
    st.session_state.pop(SESSION_KEYS["user"], None)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_check() -> bool:
    """Return True if the backend is reachable (no auth required)."""
    try:
        r = requests.get(f"{_base().rsplit('/api', 1)[0]}/docs", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def list_jobs(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    r = _ok(requests.get(f"{_base()}/jobs", params=params, headers=_headers()))
    return r.json()


def get_job(job_id: UUID | str) -> dict:
    r = _ok(requests.get(f"{_base()}/jobs/{job_id}", headers=_headers()))
    return r.json()


def create_job(
    title: str,
    raw_jd_text: str = "",
    structured: Optional[dict] = None,
    status: str = "active",
) -> dict:
    r = _ok(requests.post(
        f"{_base()}/jobs",
        json={
            "title": title,
            "raw_jd_text": raw_jd_text,
            "structured": structured or {},
            "status": status,
        },
        headers=_headers(),
    ))
    return r.json()


def upload_job(file_bytes: bytes, file_name: str, title: str = "") -> dict:
    """Upload a JD file. Uses multipart/form-data."""
    r = _ok(requests.post(
        f"{_base()}/jobs/upload",
        files={"file": (file_name, file_bytes)},
        data={"title": title or file_name},
        headers={"Authorization": _headers()["Authorization"]},  # no Content-Type — requests sets multipart boundary
    ))
    return r.json()


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def list_candidates(
    keyword: Optional[str] = None,
    skill: Optional[str] = None,
    education: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[dict]:
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if keyword:
        params["keyword"] = keyword
    if skill:
        params["skill"] = skill
    if education:
        params["education"] = education
    r = _ok(requests.get(f"{_base()}/candidates", params=params, headers=_headers()))
    return r.json()


def get_candidate(candidate_id: UUID | str) -> dict:
    r = _ok(requests.get(f"{_base()}/candidates/{candidate_id}", headers=_headers()))
    return r.json()


# ---------------------------------------------------------------------------
# Resumes
# ---------------------------------------------------------------------------

def list_resumes(candidate_id: Optional[UUID | str] = None) -> List[dict]:
    params = {}
    if candidate_id:
        params["candidate_id"] = str(candidate_id)
    r = _ok(requests.get(f"{_base()}/resumes", params=params, headers=_headers()))
    return r.json()


def get_resume(resume_id: UUID | str) -> dict:
    r = _ok(requests.get(f"{_base()}/resumes/{resume_id}", headers=_headers()))
    return r.json()


def upload_resume(
    file_bytes: bytes,
    file_name: str,
    candidate_id: Optional[UUID | str] = None,
) -> dict:
    """Upload a resume file. Creates a new candidate if candidate_id is not provided."""
    data: Dict[str, str] = {}
    if candidate_id:
        data["candidate_id"] = str(candidate_id)
    r = _ok(requests.post(
        f"{_base()}/resumes/upload",
        files={"file": (file_name, file_bytes)},
        data=data,
        headers={"Authorization": _headers()["Authorization"]},
    ))
    return r.json()


# ---------------------------------------------------------------------------
# Tasks (async)
# ---------------------------------------------------------------------------

def create_resume_upload_task(
    file_bytes: bytes,
    file_name: str,
    candidate_id: Optional[UUID | str] = None,
) -> dict:
    """Create an async resume upload task. Returns task object."""
    data: Dict[str, str] = {}
    if candidate_id:
        data["candidate_id"] = str(candidate_id)
    r = _ok(requests.post(
        f"{_base()}/tasks/resume-upload",
        files={"file": (file_name, file_bytes)},
        data=data,
        headers={"Authorization": _headers()["Authorization"]},
    ))
    return r.json()


def create_matching_task(job_id: UUID | str, candidate_ids: Optional[List[UUID | str]] = None) -> dict:
    """Create an async matching task."""
    body: Dict[str, Any] = {"job_id": str(job_id)}
    if candidate_ids:
        body["candidate_ids"] = [str(c) for c in candidate_ids]
    r = _ok(requests.post(
        f"{_base()}/tasks/matching-run",
        json=body,
        headers=_headers(),
    ))
    return r.json()


def get_task(task_id: UUID | str) -> dict:
    r = _ok(requests.get(f"{_base()}/tasks/{task_id}", headers=_headers()))
    return r.json()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def run_matching(
    job_id: UUID | str,
    candidate_ids: Optional[List[UUID | str]] = None,
) -> List[dict]:
    body: Dict[str, Any] = {"job_id": str(job_id)}
    if candidate_ids:
        body["candidate_ids"] = [str(c) for c in candidate_ids]
    r = _ok(requests.post(
        f"{_base()}/matching/run",
        json=body,
        headers=_headers(),
    ))
    return r.json()


def education_filter(
    job_id: UUID | str,
    candidate_ids: Optional[List[UUID | str]] = None,
) -> dict:
    body: Dict[str, Any] = {"job_id": str(job_id)}
    if candidate_ids:
        body["candidate_ids"] = [str(c) for c in candidate_ids]
    r = _ok(requests.post(
        f"{_base()}/matching/education-filter",
        json=body,
        headers=_headers(),
    ))
    return r.json()


def get_matches_by_job(
    job_id: UUID | str,
    limit: int = 200,
    offset: int = 0,
) -> List[dict]:
    r = _ok(requests.get(
        f"{_base()}/matching/job/{job_id}",
        params={"limit": limit, "offset": offset},
        headers=_headers(),
    ))
    return r.json()


def retry_match(job_id: UUID | str, candidate_id: UUID | str) -> List[dict]:
    r = _ok(requests.post(
        f"{_base()}/matching/retry-for-candidate",
        json={"job_id": str(job_id), "candidate_id": str(candidate_id)},
        headers=_headers(),
    ))
    return r.json()
