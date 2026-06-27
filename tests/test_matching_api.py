# tests/test_matching_api.py
"""API tests for matching endpoints."""

from fastapi.testclient import TestClient


def test_run_matching(client: TestClient):
    job = client.post(
        "/api/v1/jobs",
        json={
            "title": "Python开发",
            "structured": {
                "required_skills": ["Python", "FastAPI"],
                "min_years": 2,
            },
            "status": "active",
        },
    )
    assert job.status_code == 200
    job_id = job.json()["id"]
    r = client.post(
        "/api/v1/matching/run",
        json={"job_id": job_id},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_education_filter_candidates(client: TestClient):
    job = client.post(
        "/api/v1/jobs",
        json={
            "title": "测试学历筛",
            "education_requirement": "本科",
            "structured": {},
            "status": "active",
        },
    )
    assert job.status_code == 200
    job_id = job.json()["id"]
    r = client.post(
        "/api/v1/matching/education-filter",
        json={"job_id": job_id},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == job_id
    assert "total_input" in data
    assert "passed_count" in data
    assert "candidates" in data
    assert isinstance(data["candidates"], list)


def test_get_matches_by_job(client: TestClient):
    job = client.post(
        "/api/v1/jobs",
        json={"title": "测试岗", "status": "active"},
    )
    assert job.status_code == 200
    job_id = job.json()["id"]
    client.post("/api/v1/matching/run", json={"job_id": job_id})
    r = client.get(f"/api/v1/matching/job/{job_id}")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
