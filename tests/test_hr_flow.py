# tests/test_hr_flow.py
"""End-to-end flow: create job -> upload resume -> parse & bind -> run matching -> get results."""

from fastapi.testclient import TestClient


def test_hr_flow(client: TestClient):
    # 1. Create job via structured-first path
    job_r = client.post(
        "/api/v1/jobs",
        json={
            "title": "后端开发工程师",
            "structured": {
                "required_skills": ["Python", "FastAPI", "PostgreSQL"],
                "preferred_skills": ["Redis", "Docker"],
                "min_years": 3,
                "education_requirement": "本科及以上",
            },
            "status": "active",
        },
    )
    assert job_r.status_code == 200
    job_id = job_r.json()["id"]

    # 2. Upload resume (TXT) -> auto parse & bind
    resume_txt = (
        "李四\n"
        "email: lisi@test.com  手机 13900001111\n"
        "技能：Python、FastAPI、PostgreSQL、Redis\n"
        "工作经历：2020-2023 某公司 后端开发\n"
        "5年开发经验。"
    )
    upload_r = client.post(
        "/api/v1/resumes/upload",
        files={"file": ("resume.txt", resume_txt.encode("utf-8"), "text/plain")},
        data={},
    )
    assert upload_r.status_code == 200
    upload_data = upload_r.json()
    assert upload_data["status"] in ("parsed", "candidate_bound")
    candidate_id = upload_data["candidate_id"]

    # 3. Candidate has been bound
    cand_r = client.get(f"/api/v1/candidates/{candidate_id}")
    assert cand_r.status_code == 200
    cand = cand_r.json()
    assert cand.get("email") == "lisi@test.com" or cand.get("name")

    # 4. Run matching
    match_r = client.post(
        "/api/v1/matching/run",
        json={"job_id": job_id},
    )
    assert match_r.status_code == 200
    matches = match_r.json()
    assert isinstance(matches, list)

    # 5. Get match results by job
    list_r = client.get(f"/api/v1/matching/job/{job_id}")
    assert list_r.status_code == 200
    results = list_r.json()
    assert isinstance(results, list)
    if results:
        first = results[0]
        assert "overall_score" in first or first.get("candidate_id") == candidate_id
        assert "candidate_name" in first or "candidate_email" in first
