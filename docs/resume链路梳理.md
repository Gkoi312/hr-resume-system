# Resume 链路梳理（仅分析上传后发生了什么）

本文只分析系统在“上传简历文件”之后，从 HTTP 入口到解析、Candidate 回填、以及是否触发异步任务/向量索引的完整链路。同步与异步两条路径都会说明，但不展开 matching/hr-agent 的执行链路。

---

## A) 文件上传入口

1. 同步上传入口（文件与解析在同一个 API 请求里完成）
   - `POST /api/v1/resumes/upload`
   - 路径：`app/api/v1/endpoints/resumes.py:upload_resume`

2. 异步上传入口（文件先落盘，再创建后台任务）
   - `POST /api/v1/tasks/resume-upload`
   - 路径：`app/api/v1/endpoints/tasks.py:create_resume_upload_task`

---

## B) PDF / DOCX 文本抽取在哪里做

文本抽取在 `app/parsers/text_extractor.py` 完成：
- `extract_text_from_file(file_bytes, filename)`
- PDF：通过 `pypdf.PdfReader` 逐页 `page.extract_text()`
- DOCX/DOC：通过 `python-docx` 读取段落 `Document(...).paragraphs`

调用点：
- 同步：`app/api/v1/endpoints/resumes.py:upload_resume`
  - `content = await file.read()`
  - `text = extract_text_from_file(content, file.filename or file_name)`
- 异步：`app/tasks/resume_task.py:handle_resume_upload`
  - `content = path.read_bytes()`
  - `text = extract_text_from_file(content, original_name)`

---

## C) parser 在哪里参与

解析器（把纯文本转为结构化 parsed 字段）在 `app/parsers/resume_parser.py` 完成：
- `parse_resume_text(text) -> Dict[str, Any]`

参与调用点：
- 同步：`resumes.py:upload_resume`
  - `parsed = parse_resume_text(text)`
  - `resume_service.save_parsed_resume(resume.id, parsed, status=RESUME_STATUS_PARSED)`
- 异步：`resume_task.py:handle_resume_upload`
  - `parsed = parse_resume_text(text)`
  - `resume_service.save_parsed_resume(resume.id, parsed, status=RESUME_STATUS_PARSED)`

---

## D) Resume 与 Candidate 的关系是如何处理的

1. 关联关系（外键归属）
   - `resumes` 表中 `candidate_id` 外键指向 `candidates.id`
   - 每个 Resume 都属于某一个 Candidate（多份 Resume -> 同一 Candidate 的允许情况在 ORM relationship 中体现）。

2. Candidate 创建策略（保证“Candidate 是唯一画像来源”的基础）
   - `ResumeService.create_resume_record(...)`
   - 若上传时没有提供 `candidate_id`：
     - 先在 `CandidateRepository.create(...)` 创建一个占位 Candidate（字段为空）
     - 再创建 Resume，并把该 Candidate 的 `id` 写入 `resumes.candidate_id`

3. parsed -> candidate 的回填策略（Candidate 字段优先、不覆盖已有画像）
   - `ResumeService.bind_candidate_from_resume(resume_id)`
   - 步骤：
     1. 从 `resume_repository.get_by_id` 取到 `resume`，前提是 `resume.parsed` 已存在
     2. 取回 `candidate_repository.get_by_id(resume.candidate_id)`
     3. 对每个 Candidate 字段执行“候选人优先”的合并：
        - 如果 Candidate 字段为空（`None / "" / [] / {}`），才用 `resume.parsed` 的对应值填入
        - 如果 Candidate 字段非空，则保留 Candidate 原值，不被 resume 覆盖

4. 回填后触发索引任务
   - `bind_candidate_from_resume` 末尾会入队 `task_type="resume_index_build"`，由 Worker 为该 Candidate 构建向量索引（见第 F 部分）。

---

## E) 为什么 README 说 Candidate 是唯一画像来源；具体代码如何保证

README 声明：上传解析结果会“回填候选人主档”，且“匹配与搜索以 Candidate 为唯一画像来源”，解析结果只用于补缺不覆盖（对应“Candidate 字段优先”的合并策略）。

代码保证点主要来自两块：

1. 匹配/评分只基于 Candidate，不直接以 Resume 为画像来源
   - `matching_service._build_candidate_profile(candidate)` 只读取 `CandidateModel` 的：
     - `skills / years_of_experience / highest_education / education / work_experience / projects / summary`
   - matching 的语义部分也是通过 `vector_store` 从“job profile 向量 + candidate profile 向量/证据 chunk”读语义证据。

2. Resume 只作为“补齐候选人画像”的输入源，真正的主画像落在 Candidate
   - `ResumeService.create_resume_record`：确保每个 Resume 归属到一个 Candidate 主档。
   - `ResumeService.bind_candidate_from_resume`：以 `_is_empty()` 判断 Candidate 字段是否为空；非空则不覆盖，避免“多份简历互相污染画像”。

---

## F) 是否会触发 task / rag index

会触发。

1. 上传完成后（同步或异步，都会走到 bind）
   - `ResumeService.bind_candidate_from_resume`：
     - `await task_service.create_task(TaskCreate(task_type="resume_index_build", resource_type="resume", resource_id=resume.id, payload={"resume_id": str(resume.id)}))`

2. Worker 执行 `resume_index_build`
   - `app/tasks/rag_index_task.py:handle_resume_index_build`
   - 逻辑：读取 `resume_id` -> 找到该 resume 的 `candidate_id`
   - 再调用 `_index_candidate(candidate_id=resume.candidate_id, resume_id=resume.id)`

3. `_index_candidate` 在 RAG 层实际构建/落库的内容
   - 构建候选人语义文本：`build_candidate_semantic_text(profile)`
   - 构建候选人证据 chunks：`build_candidate_chunks(profile)`
   - 生成 embedding：`embedding_client.embed_texts(...)`
   - 落库：
     - `vector_store.upsert_candidate_profile(...)` 写入 `vector_profiles`（candidate 的 general / chunks profile）
     - `vector_store.upsert_resume_chunks(...)` 写入 `resume_chunk_embeddings`（以 candidate 为主键，resume_id 用于标记证据来源）

---

## G) 同步路径和异步路径的区别

1. 同步路径：文件上传与解析在 API 进程内完成
   - `endpoints/resumes.py:upload_resume`
   - 流程：create_resume_record -> update_status(EXTRACTING) -> extract_text_from_file -> parse_resume_text -> save_parsed_resume -> bind_candidate_from_resume -> update_status(CANDIDATE_BOUND)
   - 返回：直接返回 `resume_service.get_resume(resume.id)`（或 resume 本身）

2. 异步路径：API 只负责入队任务，真正解析发生在 Worker
   - `endpoints/tasks.py:create_resume_upload_task`
     - 把上传文件保存到磁盘（`RESUME_UPLOAD_DIR`，默认 `uploads/`）
     - 创建 `task_type="resume_upload"`，payload 包含 `file_path/original_name/candidate_id`
   - Worker 侧：`tasks/resume_task.py:handle_resume_upload`
     - 标记 task running
     - 读取落盘文件 bytes
     - extract_text_from_file + parse_resume_text
     - save_parsed_resume + bind_candidate_from_resume + update_resume_status(CANDIDATE_BOUND)
   - 返回：`/tasks/{id}` 由调用方轮询任务状态（结果在 task 的 `result_summary` 与 resume 表的状态里体现）

---

## H) 主调用链（按同步/异步分别列出）

### H1) 同步：`POST /api/v1/resumes/upload`
1. `app/api/v1/endpoints/resumes.py:upload_resume`
2. `app/services/resume_service.py:ResumeService.create_resume_record`
3. `app/services/resume_service.py:ResumeService.update_resume_status(RESUME_STATUS_EXTRACTING)`
4. `app/parsers/text_extractor.py:extract_text_from_file`
5. `app/parsers/resume_parser.py:parse_resume_text`
6. `app/services/resume_service.py:ResumeService.save_parsed_resume(RESUME_STATUS_PARSED)`
7. `app/services/resume_service.py:ResumeService.bind_candidate_from_resume`
8. `app/services/resume_service.py:ResumeService.update_resume_status(RESUME_STATUS_CANDIDATE_BOUND)`
9. `app/services/resume_service.py` 内部会入队 `task_type="resume_index_build"`
10. Worker 执行 `app/tasks/rag_index_task.py:handle_resume_index_build`

### H2) 异步：`POST /api/v1/tasks/resume-upload`
1. `app/api/v1/endpoints/tasks.py:create_resume_upload_task`
2. `app/services/task_service.py:TaskService.create_task` -> `TaskRepository.create`
3. Worker 执行 `app/tasks/resume_task.py:handle_resume_upload`
4. `ResumeService.create_resume_record`
5. `extract_text_from_file` + `parse_resume_text`
6. `save_parsed_resume` + `bind_candidate_from_resume`
7. `task_service.mark_completed`
8. Worker 后续执行 `app/tasks/rag_index_task.py:handle_resume_index_build`（由 bind 入队）

---

## I) 关键对象变化过程（核心状态/字段变化）

下面用“对象->状态->关键字段变化”的方式描述关键节点：

1. Candidate（画像主档）
   - 若不存在 candidate_id：
     - 创建：`candidates` 写入空字段（`highest_education` 通常为空，后续在 `bind_candidate_from_resume` 补齐）
   - 若存在：
     - 不覆盖非空字段，只对空字段用 parsed 补齐

2. Resume（上传记录 + parsed）
   - create：`resumes.status = RESUME_STATUS_UPLOADED`，`parsed=None`
   - extracting：`resumes.status = RESUME_STATUS_EXTRACTING`
   - parsed：`resumes.parsed = parsed_dict`，`resumes.status = RESUME_STATUS_PARSED`
   - bind 完成：`resumes.status = RESUME_STATUS_CANDIDATE_BOUND`
   - 失败：`resumes.status = RESUME_STATUS_FAILED` + `error_message=str(exc)`

3. Tasks（后台异步索引）
   - bind 后入队 `resume_index_build`：
     - `tasks.status`：pending -> running -> completed/failed
   - 若走异步上传，还会额外存在 `resume_upload` task：
     - API 创建 -> Worker 运行 -> completed/failed

---

## J) 涉及的数据表

上传与回填直接涉及：
1. `resumes`
   - 字段：`candidate_id / file_path / file_name / parsed / status / error_message`
2. `candidates`
   - 字段：`name / email / phone / education / work_experience / skills / projects / years_of_experience / summary / highest_education`

异步索引相关涉及：
3. `tasks`
   - 字段：`task_type / resource_type / resource_id / payload / status / error_message / result_summary`
4. `vector_profiles`
   - 写入对象：candidate 的 `profile_type="general"` 与 `profile_type="chunks"`（由 `_index_candidate` 决定）
   - 字段：`entity_type/entity_id/profile_type/vector/meta/content_hash/embedding_model/status/error_message`
5. `resume_chunk_embeddings`
   - 写入对象：candidate 的 evidence chunks（以 `resume_id` 标记 chunk 来源）
   - 字段：`candidate_id / resume_id / source_type / text / chunk_index / vector / importance_weight / time_weight / content_hash / embedding_model / status`

---

## K) resume -> candidate 的关系说明（只讲“谁是主画像”）

- `resumes` 是“文件载体 + 解析结果快照”（`resumes.parsed` 保存结构化解析字典）。
- `candidates` 是“唯一画像主档”（matching 与语义证据构建以 Candidate 字段为准）。
- `ResumeService.bind_candidate_from_resume` 实现了“parsed 只补缺不覆盖”的合并策略：
  - Candidate 非空字段优先保留
  - 只有 Candidate 为空时才从 Resume.parsed 填入

因此同一个 Candidate 可以有多份 Resume，但最终用于匹配/检索的画像会聚合在 Candidate 主档里。

