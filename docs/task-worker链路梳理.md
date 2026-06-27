# Task / Worker 链路梳理

本文件只分析“任务表 `tasks` + Worker 调度 + Task handler 执行 + 回调业务 Service”的链路。重点回答异步化如何发生、并发如何保证、以及 Task handler 具体如何复用既有业务（service / rag / parser）。

---

## 1) 哪些业务会写入 `tasks` 表

写入 `tasks` 的入口都走 `TaskService.create_task(...) -> TaskRepository.create(...)`，因此本质上都是“创建一条 pending 工单”。

### 1.1 HTTP 入口（由调用方显式发起异步）

1. `app/api/v1/endpoints/tasks.py:create_resume_upload_task`
   - `task_type="resume_upload"`
   - `payload={"file_path","original_name","candidate_id"}`

2. `app/api/v1/endpoints/tasks.py:create_matching_task`
   - `task_type="matching_run"`
   - `payload={"job_id","candidate_ids"}`

### 1.2 Service 内部（由业务事件触发“最佳努力”索引任务）

1. `app/services/resume_service.py:bind_candidate_from_resume`
   - 在完成“resume parsed -> candidate 主档补齐”后
   - `task_type="resume_index_build"`
   - `payload={"resume_id": str(resume.id)}`

2. `app/services/candidate_service.py:create_candidate`
   - 创建 candidate 后入队索引
   - `task_type="candidate_profile_index_build"`
   - `payload={"candidate_id": str(candidate.id)}`

3. `app/services/job_service.py:create_job / update_job`
   - job（`structured` 或 `raw_jd_text`）存在时入队索引
   - `task_type="job_profile_index_build"`
   - `payload={"job_id": str(job.id)}`

4. `app/services/job_service.py:retry_analyze_job`
   - `raw_jd_text -> structured` 解析完成后入队索引
   - `task_type="job_profile_index_build"`

---

## 2) Task 的类型有哪些

Worker 的 dispatch 分支与轮询过滤共同定义了当前 task 类型集合：

1. `matching_run`：`app/tasks/match_task.py:handle_matching_run`
2. `resume_upload`：`app/tasks/resume_task.py:handle_resume_upload`
3. `job_profile_index_build`：`app/tasks/rag_index_task.py:handle_job_profile_index_build`
4. `candidate_profile_index_build`：`app/tasks/rag_index_task.py:handle_candidate_profile_index_build`
5. `resume_index_build`：`app/tasks/rag_index_task.py:handle_resume_index_build`

---

## 3) Worker 的主循环和 dispatch 逻辑在哪里

### 3.1 主循环
文件：`app/workers/task_worker.py`

关键逻辑：
1. 初始化 DB：`await init_db()`
2. 创建并发控制信号量：`sem = asyncio.Semaphore(MAX_CONCURRENCY)`
3. 无限轮询：
   - `task = await task_repository.acquire_next_pending(task_types=[...])`
   - 若无 pending：`await asyncio.sleep(POLL_INTERVAL_SECONDS)`
4. 每个任务创建一个协程 `_run(t)`：
   - 进入 `async with sem` 控制并发
   - 成功：`dispatch_task(t)` 后记录 completed
   - 失败：`task_service.mark_failed(...)`

对应代码位置：
- `worker_loop()`：主循环
- `dispatch_task(task)`：根据 `task.task_type` 分发到具体 handler

### 3.2 dispatch
文件：`app/workers/task_worker.py`

dispatch 映射关系：
- `matching_run` -> `handle_matching_run`
- `resume_upload` -> `handle_resume_upload`
- `job_profile_index_build` -> `handle_job_profile_index_build`
- `candidate_profile_index_build` -> `handle_candidate_profile_index_build`
- `resume_index_build` -> `handle_resume_index_build`

---

## 4) acquire_next_pending / SKIP LOCKED 是怎么实现并发安全的

文件：`app/database/repository/task_repository.py:TaskRepository.acquire_next_pending`

实现要点：
1. 只选择 pending：
   - `where(TaskModel.status == TASK_STATUS_PENDING)`
2. 按创建时间升序取一个：
   - `order_by(TaskModel.created_at.asc()).limit(1)`
3. 并发关键：对该 pending 行加行锁并跳过已被其他事务锁定的行：
   - `with_for_update(skip_locked=True)`

因此多实例 Worker 可以同时运行：
1. Worker A 对某条 pending 加锁并将其置为 running
2. Worker B 使用 SKIP LOCKED 跳过被锁的行，去取下一条 pending
3. 避免同一 task 被多个 Worker 重复执行

事务边界：
- `acquire_next_pending` 使用 `async with get_session_context() as session`
- `get_session_context()` 在成功时 commit，在异常时 rollback

---

## 5) Task 如何调用原来的 service / rag / parser 逻辑

Task handler 本质上是“把原本同步/业务层的逻辑在 Worker 里执行”，并不重写业务算法。

### 5.1 `resume_upload` -> parser + resume_service + bind + 索引入队
文件：`app/tasks/resume_task.py:handle_resume_upload`
1. 从任务 payload 获取磁盘文件路径
2. 创建 resume：`resume_service.create_resume_record`
3. 标记 extracting：`resume_service.update_resume_status`
4. 文本抽取：
   - `extract_text_from_file(...)`（PDF/DOCX/TXT）
5. 结构化解析：
   - `parse_resume_text(text)`（resume_parser）
6. 保存 parsed：
   - `resume_service.save_parsed_resume`
7. 绑定 candidate（Candidate 字段优先补齐）：
   - `resume_service.bind_candidate_from_resume`
8. handler 自身不会直接构建向量，而是依赖 bind 内部入队 `resume_index_build` task

### 5.2 `matching_run` -> matching_service
文件：`app/tasks/match_task.py:handle_matching_run`
1. 解析 payload：`job_id` 与可选 `candidate_ids`
2. 执行匹配：
   - `matching_service.run_matching(job_id, candidate_ids)`
3. 完成后写入 task 的 result_summary（match_count）

matching_service 内部会调用：
- 规则四维评分函数
- semantic 维度的 vector_store 检索（job/candidate/vector profile + resume chunks 证据）
- `match_repository.create` 写入 `candidate_job_matches`

### 5.3 `*_index_build` -> rag_index_task -> embedding + vector_store + profile_builder
文件：`app/tasks/rag_index_task.py`
- `handle_job_profile_index_build`
  - `job_repository.get_by_id`
  - `build_job_semantic_text`（profile_builder）
  - `stable_hash_text`
  - `embedding_client.embed_texts`
  - `vector_store.upsert_job_profile`（写 `vector_profiles`）
- `handle_candidate_profile_index_build`
  - `candidate_repository.get_by_id`
  - `build_candidate_semantic_text`
  - `build_candidate_chunks`
  - embedding
  - `vector_store.upsert_candidate_profile` + `vector_store.upsert_resume_chunks`（写向量与证据表）
- `handle_resume_index_build`
  - `resume_repository.get_by_id(resume_id)`
  - 用 `resume.candidate_id` 调 `_index_candidate(candidate_id, resume_id=resume.id)`
  - 写入 `vector_profiles` / `resume_chunk_embeddings`

---

## 6) Task 是不是只是改变执行时机，而不是改变业务分层

是。

从代码组织看：
1. 业务分层仍然是 `API -> Service -> Repository`（同步）
2. 引入 task/worker 后变为 `API/Service -> TaskService/TaskRepository -> Worker -> Task handler -> 调用原 Service/业务组件`
3. Task handler 只负责：
   - 从 tasks 表拿到 payload
   - 调用相同的 service / parser / rag 组件
   - 更新 tasks 状态（running/completed/failed）

因此算法与数据治理仍在原有 service 与 rag/parser 内，而 task/worker 只提供异步执行与并发调度。

---

## 7) Task 生命周期（文字版）

以 `resume_upload` 为例（其它 task 类似）：
1. 任务创建：`TaskService.create_task(...)`
   - `tasks.status = pending`
2. Worker 获取：`task_repository.acquire_next_pending(...)`
   - 行锁 + SKIP LOCKED
   - 将 `tasks.status` 从 pending -> running
3. Worker dispatch 到 handler：
   - handler 可能再调用 `task_service.mark_running(...)`（resume_task/matching_run/index build handler 会显式 mark_running）
4. handler 成功：
   - 更新业务表（resumes/candidates 或 match_repository 等）
   - `task_service.mark_completed(..., result_summary=...)`
5. handler 失败：
   - `task_service.mark_failed(task.id, str(exc))`

---

## 8) Task 与同步 API 的关系

1. `/api/v1/resumes/upload`：
   - 同步执行 extract/parse/bind，并在 bind 后入队 `resume_index_build`
2. `/api/v1/tasks/resume-upload`：
   - 异步执行同样的 extract/parse/bind（但先把文件落到 `RESUME_UPLOAD_DIR`）
3. `/api/v1/matching/run`：
   - 同步调用 `matching_service.run_matching` 直接落库 matches
4. `/api/v1/tasks/matching-run`：
   - 异步入队 `matching_run`，Worker 执行同样的 `matching_service.run_matching`

换言之：
- 同步 API 把耗时工作留在请求线程
- Task API 把耗时工作转移到 Worker，但仍调用同一套 service/rag/parser 逻辑

---

## 9) 这条链在架构上的作用

1. 解耦 API 响应时间与耗时计算
   - 简历解析/向量索引/全量匹配都可能耗时
2. 提供“多实例 Worker 可扩展”的并发执行能力
   - `SELECT ... FOR UPDATE SKIP LOCKED` 防止重复消费
3. 形成统一的异步工单状态机
   - pending -> running -> completed/failed
4. 将“RAG 索引构建”与“业务事件”绑定
   - 例如 resume/candidate/job 的写入完成后，通过入队 index build 让 semantic profile 持久化到向量表

