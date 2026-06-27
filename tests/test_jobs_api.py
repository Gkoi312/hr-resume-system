# tests/test_jobs_api.py
"""API tests for jobs endpoints (structured-first + upload)."""

from fastapi.testclient import TestClient

from app.parsers.job_parser.pipeline import LLMParseError, OCRParseError, UnsupportedFileTypeError


def test_create_job_structured(client: TestClient):
    r = client.post(
        "/api/v1/jobs",
        json={
            "title": "后端开发工程师",
            "structured": {
                "required_skills": ["Python", "FastAPI"],
                "preferred_skills": ["Redis"],
                "min_years": 3,
                "education_requirement": "本科及以上",
            },
            "status": "active",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "后端开发工程师"
    assert data["status"] == "active"
    assert data.get("structured") is not None


def test_get_and_list_jobs(client: TestClient):
    create = client.post("/api/v1/jobs", json={"title": "测试岗位", "status": "active"})
    assert create.status_code == 200
    job_id = create.json()["id"]

    get_r = client.get(f"/api/v1/jobs/{job_id}")
    assert get_r.status_code == 200
    assert get_r.json()["id"] == job_id

    list_r = client.get("/api/v1/jobs", params={"limit": 10, "offset": 0})
    assert list_r.status_code == 200
    assert isinstance(list_r.json(), list)


def test_update_job_structured(client: TestClient):
    create = client.post("/api/v1/jobs", json={"title": "原标题", "status": "active"})
    assert create.status_code == 200
    job_id = create.json()["id"]
    r = client.patch(
        f"/api/v1/jobs/{job_id}",
        json={
            "title": "新标题",
            "structured": {"required_skills": ["Python", "SQL"]},
        },
    )
    assert r.status_code == 200
    assert r.json()["title"] == "新标题"
    assert "required_skills" in (r.json().get("structured") or {})


def test_upload_job_uses_parser_pipeline(client: TestClient, monkeypatch):
    async def _fake_parse_job_document(file_bytes: bytes, file_name: str):
        return {
            "raw_jd_text": "岗位：Python工程师",
            "structured": {
                "job_title": "Python工程师",
                "required_skills": ["Python", "FastAPI"],
                "min_years": 2,
            },
            "text_extraction_method": "fake",
        }

    monkeypatch.setattr(
        "app.api.v1.endpoints.jobs.parse_job_document",
        _fake_parse_job_document,
    )

    r = client.post(
        "/api/v1/jobs/upload",
        files={"file": ("jd.txt", "这是JD".encode("utf-8"), "text/plain")},
        data={"status": "active"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Python工程师"
    assert body["raw_jd_text"] == "岗位：Python工程师"
    assert (body.get("structured") or {}).get("required_skills") == ["Python", "FastAPI"]


def test_upload_job_unsupported_file_type(client: TestClient, monkeypatch):
    async def _raise_unsupported(file_bytes: bytes, file_name: str):
        raise UnsupportedFileTypeError(file_name)

    monkeypatch.setattr(
        "app.api.v1.endpoints.jobs.parse_job_document",
        _raise_unsupported,
    )
    r = client.post(
        "/api/v1/jobs/upload",
        files={"file": ("jd.exe", b"fake", "application/octet-stream")},
    )
    assert r.status_code == 415
    assert r.json()["detail"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_upload_job_ocr_failed(client: TestClient, monkeypatch):
    async def _raise_ocr_failed(file_bytes: bytes, file_name: str):
        raise OCRParseError("PaddleOCR failed: timeout")

    monkeypatch.setattr(
        "app.api.v1.endpoints.jobs.parse_job_document",
        _raise_ocr_failed,
    )
    r = client.post(
        "/api/v1/jobs/upload",
        files={"file": ("jd.pdf", b"%PDF-1.7", "application/pdf")},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "OCR_FAILED"


def test_upload_job_llm_failed(client: TestClient, monkeypatch):
    async def _raise_llm_failed(file_bytes: bytes, file_name: str):
        raise LLMParseError("LLM provider unavailable")

    monkeypatch.setattr(
        "app.api.v1.endpoints.jobs.parse_job_document",
        _raise_llm_failed,
    )
    r = client.post(
        "/api/v1/jobs/upload",
        files={"file": ("jd.txt", "这是JD".encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "LLM_FAILED"

