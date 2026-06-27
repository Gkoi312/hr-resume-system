# tests/test_jobs_auth.py
"""Jobs API with HR authentication: ownership and list scoping."""

import uuid

from fastapi.testclient import TestClient


def _auth_header(client: TestClient, username: str, password: str) -> dict[str, str]:
    r = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert r.status_code == 200
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_job_list_scoped_to_current_user(client: TestClient):
    sfx = uuid.uuid4().hex[:10]
    ua = f"hr_jobs_a_{sfx}"
    ub = f"hr_jobs_b_{sfx}"
    assert client.post(
        "/api/v1/auth/register",
        json={"username": ua, "password": "secret12"},
    ).status_code == 200
    assert client.post(
        "/api/v1/auth/register",
        json={"username": ub, "password": "secret12"},
    ).status_code == 200
    ha = _auth_header(client, ua, "secret12")
    hb = _auth_header(client, ub, "secret12")

    ja = client.post(
        "/api/v1/jobs",
        json={"title": "Job A", "status": "active"},
        headers=ha,
    )
    jb = client.post(
        "/api/v1/jobs",
        json={"title": "Job B", "status": "active"},
        headers=hb,
    )
    assert ja.status_code == 200 and jb.status_code == 200

    list_a = client.get("/api/v1/jobs", headers=ha)
    assert list_a.status_code == 200
    titles_a = {j["title"] for j in list_a.json()}
    assert "Job A" in titles_a
    assert "Job B" not in titles_a


def test_job_forbidden_for_other_user(client: TestClient):
    sfx = uuid.uuid4().hex[:10]
    uo = f"hr_owner_{sfx}"
    ui = f"hr_intruder_{sfx}"
    assert client.post(
        "/api/v1/auth/register",
        json={"username": uo, "password": "secret12"},
    ).status_code == 200
    assert client.post(
        "/api/v1/auth/register",
        json={"username": ui, "password": "secret12"},
    ).status_code == 200
    ho = _auth_header(client, uo, "secret12")
    hi = _auth_header(client, ui, "secret12")

    create = client.post(
        "/api/v1/jobs",
        json={"title": "Owned job", "status": "active"},
        headers=ho,
    )
    assert create.status_code == 200
    job_id = create.json()["id"]

    g = client.get(f"/api/v1/jobs/{job_id}", headers=hi)
    assert g.status_code == 403
