# tests/test_resumes_api.py
"""API tests for resumes endpoints."""

from fastapi.testclient import TestClient

from app.parsers.resume_parser.candidate_profile_builder import get_candidate_bind_for_resume


def test_upload_resume_no_file(client: TestClient):
    """Create resume record without file (form: file_name only)."""
    r = client.post(
        "/api/v1/resumes/upload",
        data={"file_name": "test.pdf"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["file_name"] == "test.pdf"
    assert data["status"] == "uploaded"
    assert "id" in data
    assert "candidate_id" in data


def test_upload_resume_txt_and_parse(client: TestClient):
    """Upload TXT resume and trigger parse + bind."""
    body = (
        "张三\n"
        "邮箱：zhangsan@example.com  电话：13800138000\n"
        "技能：Python、FastAPI、PostgreSQL\n"
        "3年开发经验。"
    )
    r = client.post(
        "/api/v1/resumes/upload",
        files={"file": ("resume.txt", body.encode("utf-8"), "text/plain")},
        data={},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("parsed", "candidate_bound")
    assert data.get("parsed") is not None
    parsed = data["parsed"]
    assert parsed.get("parser_version") == "resume_v1"
    assert "layer_1_extracted" in parsed
    assert "layer_2_normalized" not in parsed
    bind = get_candidate_bind_for_resume(parsed)
    assert bind.get("email") == "zhangsan@example.com" or bind.get("name")
    # Bind should have run: candidate has name/email
    candidate_id = data["candidate_id"]
    cr = client.get(f"/api/v1/candidates/{candidate_id}")
    assert cr.status_code == 200
    cand = cr.json()
    assert cand.get("email") == "zhangsan@example.com" or cand.get("name")


def test_get_resume(client: TestClient):
    create = client.post(
        "/api/v1/resumes/upload",
        data={"file_name": "x.pdf"},
    )
    assert create.status_code == 200
    resume_id = create.json()["id"]
    r = client.get(f"/api/v1/resumes/{resume_id}")
    assert r.status_code == 200
    assert r.json()["id"] == resume_id


def test_list_resumes(client: TestClient):
    create = client.post(
        "/api/v1/resumes/upload",
        data={"file_name": "y.pdf"},
    )
    assert create.status_code == 200
    candidate_id = create.json()["candidate_id"]
    r = client.get("/api/v1/resumes", params={"candidate_id": candidate_id})
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) >= 1


def test_bind_endpoint_updates_status_candidate_bound(client: TestClient):
    body = (
        "王五\n"
        "邮箱：wangwu@example.com  电话：13800138001\n"
        "技能：Python、FastAPI\n"
        "2年经验。"
    )
    up = client.post(
        "/api/v1/resumes/upload",
        files={"file": ("resume.txt", body.encode("utf-8"), "text/plain")},
        data={},
    )
    assert up.status_code == 200
    resume_id = up.json()["id"]

    # Call bind explicitly (should be idempotent) and ensure status reaches candidate_bound.
    br = client.post(f"/api/v1/resumes/{resume_id}/bind")
    assert br.status_code == 200
    assert br.json()["status"] == "candidate_bound"


def test_retry_parse_clears_parsed_and_error(client: TestClient):
    body = (
        "赵六\n"
        "email: zhaoliu@example.com  手机 13800138002\n"
        "技能：Python\n"
        "1年经验。"
    )
    up = client.post(
        "/api/v1/resumes/upload",
        files={"file": ("resume.txt", body.encode("utf-8"), "text/plain")},
        data={},
    )
    assert up.status_code == 200
    resume_id = up.json()["id"]

    # Ensure parsed exists before retry.
    before = client.get(f"/api/v1/resumes/{resume_id}")
    assert before.status_code == 200
    assert before.json().get("parsed") is not None

    rr = client.post(f"/api/v1/resumes/{resume_id}/retry-parse")
    assert rr.status_code == 200
    after = rr.json()
    assert after["status"] == "uploaded"
    assert after.get("parsed") is None
    assert after.get("error_message") is None
