# Job 链路梳理（仅分析 job）

本文档只分析 Job（岗位/JD）从 API 进入系统，到落库并触发可选的向量索引构建的链路；不展开 resume/matching/hr-agent 的完整执行流程。

---

## 1. Job 从哪个 API 入口进入

Job 的“写入/触发索引”主要从以下三个 HTTP 入口进入：

1. `POST /api/v1/jobs`（`app/api/v1/endpoints/jobs.py:create_job`）
2. `PATCH /api/v1/jobs/{job_id}`（`app/api/v1/endpoints/jobs.py:update_job`）
3. `POST /api/v1/jobs/{job_id}/retry-analyze`（`app/api/v1/endpoints/jobs.py:retry_analyze_job`）

说明（2026-03 起）：
- 当 `auto_analyze=true` 且存在 `raw_jd_text`、尚未提供 `structured` 时，`JobService.create_job` / `update_job` 会**同步**调用 `app/parsers/jd_analyzer.py:analyze_jd_text` 写入 `structured`；成功则 `status=active`，失败则 `status=failed` 并写入 `error_message`。
- `retry_analyze_job` 仍可从 `raw_jd_text` 重试结构化（异步语义不变，仍在 service 内同步执行分析后落库）。

### 1.1 与 HR 用户（JWT）

- `POST/GET/PATCH` jobs 及含 `job_id` 的 matching / tasks / hr-agent 接口支持可选 `Authorization: Bearer`。
- 已登录时：创建 job 会强制 `created_by_id=当前用户`；列表仅返回当前用户的岗位；访问他人岗位返回 403。
- 未登录时：行为与早期版本一致（列表不按用户过滤，不设创建人）。

---

## 2. Endpoint 调了哪个 Service

1. `create_job` -> `app/services/job_service.py:JobService.create_job`
2. `update_job` -> `app/services/job_service.py:JobService.update_job`
3. `retry_analyze_job` -> `app/services/job_service.py:JobService.retry_analyze_job`

---

## 3. Service 又调用了哪些 repository / parser / rag 组件

### 3.1 `create_job` / `update_job` 路径（可选 `jd_analyzer`）

`JobService.create_job / update_job` 调用链：

1. （条件满足时）`app/parsers/jd_analyzer.py:analyze_jd_text`
2. `app/database/repository/job_repository.py:JobRepository.create/update`
2. （落库后，满足条件时）`app/services/task_service.py:TaskService.create_task`
3. 任务由 Worker 执行：
   - `app/tasks/rag_index_task.py:handle_job_profile_index_build`
   - 该 handler 内部调用：
     - `app/database/repository/job_repository.py:JobRepository.get_by_id`
     - `app/rag/profile_builder.py:build_job_semantic_text`
     - `app/rag/profile_builder.py:stable_hash_text`
     - `app/llm/embedding.py:embedding_client.embed_texts`
     - `app/rag/vector_store.py:vector_store.upsert_job_profile`（DB 落库）

### 3.2 `retry_analyze_job` 路径（包含 `jd_analyzer`）

`JobService.retry_analyze_job` 调用链：

1. `JobRepository.get_by_id`
2. `JobRepository.update`（`status=JOB_STATUS_ANALYZING`，清 `error_message`）
3. `app/parsers/jd_analyzer.py:analyze_jd_text`（生成 `structured` 字段）
4. `JobRepository.update`（写 `structured`，`status=JOB_STATUS_ACTIVE`）
5. （成功后）`TaskService.create_task` 入队 `job_profile_index_build`

随后同样进入 Worker 的 `handle_job_profile_index_build`，完成向量落库。

---

## 4. 最终写入哪些表、哪些字段

job 链路会写入两类表：业务表（`jobs`）与基础设施表（`tasks`、`vector_profiles`）。

### 4.1 `jobs` 表（岗位主档）

由 `app/database/repository/job_repository.py` 写入。

写入/更新字段（由代码显式赋值决定）：
- `title`
- `created_by_id`（create 路径）
- `raw_jd_text`
- `structured`
- `status`
- `error_message`（update 路径在某些分支显式写入；create 路径不写）

字段补充说明：
- `id` 由 ORM 默认生成（`default=uuid.uuid4`）。
- `updated_at` 由数据库/SQLAlchemy onupdate 行为自动刷新（代码不显式赋值）。

### 4.2 `tasks` 表（异步任务状态）

由 `app/services/task_service.py` + `app/database/repository/task_repository.py` 写入。

写入字段：
- 创建时：`task_type`、`resource_type`、`resource_id`、`payload`、`status=TASK_STATUS_PENDING`
- Worker 执行中：
  - `status=TASK_STATUS_RUNNING`，并清空 `error_message=None`
  - 成功：`status=TASK_STATUS_COMPLETED`，写 `result_summary`
  - 失败：`status=TASK_STATUS_FAILED`，写 `error_message`

对应的任务类型：
- `task_type="job_profile_index_build"`
- `resource_type="job"`
- `payload={"job_id": str(job.id)}`

### 4.3 `vector_profiles` 表（Job 的向量索引）

由 `app/rag/vector_store.py:DbVectorStore.upsert_job_profile` 最终写入 `app/database/models.py:VectorProfileModel`。

写入字段（以成功与失败分支为边界）：
- `entity_type="job"`
- `entity_id=job.id`
- `profile_type="general"`
- `vector`
  - 成功：写入 `job_vec`（embedding_client 输出的向量）
  - 失败：写入空列表 `[]`
- `meta`：`{"semantic_profile_text": job_text}`
- `content_hash=stable_hash_text(job_text)`
- `embedding_model=embedding_client.model_name`（可能为 None）
- `status`
  - 成功：`"available"`
  - embedding 失败：`"embedding_failed"`
- `error_message`
  - embedding 失败：写入异常字符串

---

## 5. 是否会触发异步任务或索引构建

会触发“索引构建”的异步任务，具体取决于 job 的字段：

1. `create_job` / `update_job`：只要 `job.structured` 或 `job.raw_jd_text` 存在，就会 `TaskService.create_task` 入队 `job_profile_index_build`。
2. `retry_analyze_job`：会写入 `structured` 后，再入队 `job_profile_index_build`。

Worker 执行 `handle_job_profile_index_build` 时会完成：
- `build_job_semantic_text` -> embedding 向量生成
- `vector_store.upsert_job_profile` -> 写入 `vector_profiles`
- 若向量记录已存在且 `content_hash` 与 `embedding_model` 未变，会走“跳过”分支（仍会更新 task 为 completed）。

---

## 6. 这条链与 matching / hr_agent 的关系

job 链路本身只负责：
- 维护岗位主档（`jobs` + `structured`）
- 构建岗位语义向量索引（`vector_profiles`，供后续语义评分检索）

与匹配/问答的依赖关系（只说明依赖点，不展开其执行链路）：
- `matching`：语义分数使用 `vector_store.get_job_profile(job.id)` 与 `job.structured`（或 job 的语义文本）共同参与。没有 job 的 vector 索引时，语义分会退化（不保证语义证据可用）。
- `hr_agent`：grounded 的“岗位解释/摘要”会依赖 `jobs` 中的 `structured` 字段与/或 `raw_jd_text`；同时，hr-agent 的解释型回答通常还会读取匹配结果/解释（这些由 matching 写入）。

因此：
- job 链路是 matching/hr-agent 的上游“数据供给者 + 语义索引提供者”。
- job 链路不会直接触发 matching 或 hr-agent 生成匹配表/回答内容。

---

## 7. 调用链（函数/类/文件路径）

### 7.1 create / update（含异步索引构建）

HTTP：
- `app/api/v1/endpoints/jobs.py:create_job`（或 `update_job`）
  -> `app/services/job_service.py:JobService.create_job`（或 `update_job`）
     -> `app/database/repository/job_repository.py:JobRepository.create`（或 `update`）
     -> `app/services/task_service.py:TaskService.create_task`
        ->（DB写入）`app/database/repository/task_repository.py:TaskRepository.create`
     （异步）Worker：
     -> `app/workers/task_worker.py:dispatch_task`
        -> `app/tasks/rag_index_task.py:handle_job_profile_index_build`
           -> `app/database/repository/job_repository.py:JobRepository.get_by_id`
           -> `app/rag/profile_builder.py:build_job_semantic_text`
           -> `app/rag/profile_builder.py:stable_hash_text`
           -> `app/llm/embedding.py:embedding_client.embed_texts`
           -> `app/rag/vector_store.py:DbVectorStore.upsert_job_profile`
           ->（写入）`app/database/models.py:VectorProfileModel`（`vector_profiles`）
           -> `app/services/task_service.py:TaskService.mark_running/mark_completed`（更新 tasks）

### 7.2 retry-analyze（先写 structured，再异步索引）

- `app/api/v1/endpoints/jobs.py:retry_analyze_job`
  -> `app/services/job_service.py:JobService.retry_analyze_job`
     -> `app/database/repository/job_repository.py:get_by_id`
     -> `app/database/repository/job_repository.py:update`（status=ANALYZING）
     -> `app/parsers/jd_analyzer.py:analyze_jd_text`（生成 structured）
     -> `app/database/repository/job_repository.py:update`（写 structured，status=ACTIVE）
     -> `app/services/task_service.py:TaskService.create_task`
     ->（异步）同 7.1 中 `handle_job_profile_index_build` 完成 vector_profiles 落库

---

## 8. 时序图（文字版）

### 8.1 `POST /api/v1/jobs` -> create_job -> 入队索引

1. Client -> `app/api/v1/endpoints/jobs.py:create_job`
2. Endpoint -> `JobService.create_job(payload)`
3. JobService -> `JobRepository.create(...)` 写入 `jobs`
4. JobService -> `TaskService.create_task(task_type="job_profile_index_build", payload={"job_id": ...})` 写入 `tasks(pending)`
5. API 返回 `JobResponse`
6. Worker -> `task_repository.acquire_next_pending()` 取出该 task
7. Worker -> `TaskService.mark_running()` 更新 `tasks(running)`
8. Worker -> `handle_job_profile_index_build`
9. Handler -> `JobRepository.get_by_id(job_id)` 读取 `jobs`
10. Handler -> `build_job_semantic_text` + `stable_hash_text` -> `embedding_client.embed_texts`
11. Handler -> `vector_store.upsert_job_profile` 写入 `vector_profiles`
12. Handler -> `TaskService.mark_completed(result_summary=...)` 更新 `tasks(completed)`

### 8.2 `POST /api/v1/jobs/{job_id}/retry-analyze` -> structured + 索引

1. Client -> `retry_analyze_job`
2. JobService -> `JobRepository.get_by_id`
3. JobService -> `JobRepository.update(status=ANALYZING, error_message=None)`
4. JobService -> `jd_analyzer.analyze_jd_text(raw_jd_text)` 得到 `structured`
5. JobService -> `JobRepository.update(structured, status=ACTIVE)`
6. JobService -> `TaskService.create_task(job_profile_index_build ...)`
7. 后续时序同 8.1 的 Worker 索引落库部分

---

## 9. 数据流（输入 -> 转换 -> 存储）

### 9.1 create/update 输入 -> 岗位落库

输入：
- `title`
- `raw_jd_text`
- `structured`（可为空）
- `status`（可由 `auto_analyze` 影响为 analyzing）

转换：
- 不做 JD 结构化解析（`structured` 若为空则保持为空；只是状态可能被推到 analyzing）

存储：
- `jobs`：写入 `title/raw_jd_text/structured/status/created_by_id/error_message(如有)`

### 9.2 create/update -> 异步索引：Job 文本 -> 向量

输入（来自任务）：
- `payload.job_id`

转换：
- Handler 读取 `jobs` 得到 `job` 与 `job.structured(若为 dict)`
- 生成 `job_text = build_job_semantic_text(job, job_structured)`
  - 组合 Job title、structured 中的字段（如 required_skills/min_years/education_requirement/industry_preference 等）以及 `raw_jd_text`
- `job_hash = stable_hash_text(job_text)`
- `job_vec = embedding_client.embed_texts([job_text])[0]`

存储：
- `vector_profiles`：
  - `vector=job_vec`
  - `meta={"semantic_profile_text": job_text}`
  - `content_hash=job_hash`
  - `status="available"`（或 embedding_failed 分支）

### 9.3 retry-analyze -> structured 补全

输入：
- `job_id`
- `raw_jd_text`（必须存在）

转换：
- `structured = analyze_jd_text(raw_jd_text)`

存储：
- `jobs.structured` 与 `jobs.status=JOB_STATUS_ACTIVE`
- 然后再走异步索引落库到 `vector_profiles`

---

## 10. 这条链在整个系统中的定位

Job 链路是系统的“语义源头”和“匹配对象定义”：
- 它把岗位描述（`raw_jd_text`）沉淀为可审计结构化字段（`structured`），并维护状态与错误原因（`jobs.status/error_message`）。
- 它把岗位沉淀为向量语义索引（`vector_profiles`），为后续 matching 的语义评分提供可检索的输入。

它不负责：
- resume 解析与候选人画像构建（这些来自 resume/candidate 链）
- matching 计算与写入 `candidate_job_matches`（来自 matching 链）
- hr-agent 问答生成（来自 hr-agent 链）

