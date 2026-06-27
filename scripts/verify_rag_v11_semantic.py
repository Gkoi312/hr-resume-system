#!/usr/bin/env python3
"""
V1.1 验证脚本：semantic_score + semantic_evidence

流程：
1) 从 jobs/backend_engineer_saas.json 创建 Job（带 structured）
2) 从 resumes/candidate_strong_match.json 创建候选人 A
3) 从 resumes/candidate_borderline_match.json 创建候选人 B
4) 调用 POST /api/v1/matching/run（candidate_ids=[A,B]）
5) 打印并做最小断言：
   - match.semantic_score 存在且为数值
   - match.explanation.semantic_evidence 为非空列表（至少 1 条）

运行前：
- 启动后端：uvicorn app.main:app --reload
- 可选：设置 BASE_URL（默认 http://127.0.0.1:8000）

用法（项目根目录）：
  python scripts/verify_rag_v11_semantic.py
  BASE_URL=http://localhost:8000 python scripts/verify_rag_v11_semantic.py
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
RESUMES_DIR = ROOT / "resumes"

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API = f"{BASE_URL}/api/v1"
OUT_PATH = os.getenv("VERIFY_OUT_PATH", str(ROOT / "verify_rag_v11_out.json"))


def req(method: str, path: str, body: dict | None = None):
    url = f"{API}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req_obj = Request(url, data=data, method=method)
    if body is not None:
        req_obj.add_header("Content-Type", "application/json")
    try:
        with urlopen(req_obj, timeout=90) as r:
            return json.loads(r.read().decode("utf-8"))
    except HTTPError as e:
        resp_body = e.read().decode("utf-8") if e.fp else ""
        print(f"HTTP {e.code} {path}: {resp_body}", file=sys.stderr)
        raise


def get(path: str):
    return req("GET", path)


def post(path: str, body: dict | None = None):
    return req("POST", path, body=body)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_candidate_from_resume_json(path: Path) -> dict:
    data = _load_json(path)
    cand = data.get("candidate") or {}
    if not cand.get("name"):
        raise ValueError(f"{path} 中 candidate.name 不能为空")
    return cand


def _find_latest_task_id(tasks: list[dict], task_type: str, resource_type: str, resource_id: str) -> str | None:
    for t in tasks:
        if not isinstance(t, dict):
            continue
        if t.get("task_type") != task_type:
            continue
        if t.get("resource_type") != resource_type:
            continue
        if str(t.get("resource_id")) != str(resource_id):
            continue
        tid = t.get("id")
        if tid:
            return str(tid)
    return None


def _wait_task_completed(task_id: str, timeout_s: int = 60) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        t = get(f"/tasks/{task_id}")
        st = t.get("status")
        if st == "completed":
            return t
        if st == "failed":
            raise AssertionError(f"task failed: {task_id}, error={t.get('error_message')}")
        time.sleep(1)
    raise AssertionError(f"task timeout: {task_id} (>{timeout_s}s). 请确认 worker 在运行。")


def main():
    print("=== V1.1/V1.2 验证 + 多 Job / 多简历匹配 ===\n")

    # 检查后端是否可用
    try:
        get("/jobs")
    except Exception as e:
        print(
            f"请先启动后端（uvicorn app.main:app --reload），并确认 BASE_URL={BASE_URL} 可访问。错误: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    backend_job_path = JOBS_DIR / "backend_engineer_saas.json"
    data_job_path = JOBS_DIR / "data_engineer_analytics.json"

    if not backend_job_path.is_file():
        print(f"未找到 {backend_job_path}", file=sys.stderr)
        sys.exit(1)

    # 1) 创建两个 Job（如果 data_job 不存在则只跑 backend）
    backend_payload = _load_json(backend_job_path)
    backend_job_body = {
        "title": backend_payload.get("title") or "后端工程师（Python / SaaS）",
        "raw_jd_text": backend_payload.get("raw_jd_text"),
        "structured": backend_payload.get("structured"),
        "status": "active",
    }
    backend_job = post("/jobs", backend_job_body)
    backend_job_id = backend_job["id"]
    print(f"[Job1] backend_engineer_saas created: {backend_job_id}")

    data_job_id: str | None = None
    if data_job_path.is_file():
        data_payload = _load_json(data_job_path)
        data_job_body = {
            "title": data_payload.get("title") or "数据工程师（Python / 数仓）",
            "raw_jd_text": data_payload.get("raw_jd_text"),
            "structured": data_payload.get("structured"),
            "status": "active",
        }
        data_job = post("/jobs", data_job_body)
        data_job_id = data_job["id"]
        print(f"[Job2] data_engineer_analytics created: {data_job_id}")
    else:
        print("未找到 jobs/data_engineer_analytics.json，将只针对后端岗位做多简历匹配。")

    # 2) 创建所有简历对应的 candidates
    print("\n=== 创建 candidates（扫描 resumes/*.json） ===")
    all_candidate_ids: list[str] = []
    for resume_path in sorted(RESUMES_DIR.glob("*.json")):
        cand_payload = _load_candidate_from_resume_json(resume_path)
        res = post("/candidates", cand_payload)
        cid = res["id"]
        all_candidate_ids.append(cid)
        print(f"- {resume_path.name}: {cid} ({cand_payload.get('name')})")

    if not all_candidate_ids:
        raise AssertionError("resumes 目录下未找到任何简历 JSON 文件")

    # 3) 等待索引任务完成（v1.2 验证）
    print("\nWaiting index build tasks (V1.2) ...")
    tasks = get("/tasks?limit=200&offset=0")
    if not isinstance(tasks, list):
        raise AssertionError("/tasks 返回非 list")

    job_ids_to_wait = [backend_job_id] + ([data_job_id] if data_job_id else [])
    for jid in job_ids_to_wait:
        tid = _find_latest_task_id(tasks, "job_profile_index_build", "job", str(jid))
        if not tid:
            raise AssertionError(f"未找到 job_profile_index_build 任务，job_id={jid}")
        _wait_task_completed(tid, timeout_s=120)

    for cid in all_candidate_ids:
        tid = _find_latest_task_id(tasks, "candidate_profile_index_build", "candidate", str(cid))
        if not tid:
            raise AssertionError(f"未找到 candidate_profile_index_build 任务，candidate_id={cid}")
        _wait_task_completed(tid, timeout_s=120)

    print("Index tasks completed.\n")

    # 4) 对两个 Job 分别跑 matching（全量候选人）
    print("Running matching for backend job with all candidates ...")
    post(
        "/matching/run",
        {"job_id": backend_job_id, "candidate_ids": all_candidate_ids},
    )
    backend_matches = get(f"/matching/job/{backend_job_id}?limit=50&offset=0")

    data_matches: list[dict] | None = None
    if data_job_id:
        print("Running matching for data engineer job with all candidates ...")
        post(
            "/matching/run",
            {"job_id": data_job_id, "candidate_ids": all_candidate_ids},
        )
        data_matches = get(f"/matching/job/{data_job_id}?limit=50&offset=0")

    # 5) 写出统一结果
    out_obj = {
        "multi_job_matching": {
            "backend_job_id": backend_job_id,
            "data_job_id": data_job_id,
            "all_candidate_ids": all_candidate_ids,
            "backend_matches": backend_matches,
            "data_engineer_matches": data_matches,
        }
    }

    try:
        Path(OUT_PATH).write_text(
            json.dumps(out_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote multi-job result to: {OUT_PATH}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to write output file {OUT_PATH}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

