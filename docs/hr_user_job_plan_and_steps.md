# HR 用户链 + Job  creation 改造：计划与执行步骤

本文档描述将 **User（HR）** 与 **Job（岗位）** 在前端主导流程下贯通的目标、分阶段计划与可执行步骤。  
现有 Job 后端与数据模型说明见：`docs/job链路梳理.md`。

---

## 1. 目标与原则

| 目标 | 说明 |
|------|------|
| User 链 | HR 具备可登录账号；创建/查看岗位默认归属当前用户，而非调用方手填 `created_by_id`。 |
| Job 主路径 | **网页表单为主**：`job_name`、必选/优选技能、学历等 → 组装为 JSON（与 `app/schemas/job.py` 中 `JobStructured` / `structured` 对齐或扩展）。 |
| Job 备选路径 | 支持上传完整 JD：**txt / word / pdf / 图片**（与简历侧能力复用或独立管道），抽到 `raw_jd_text` 后走结构化分析与索引。 |
| 一致性 | 修复或明确 **`POST /jobs` 上 `auto_analyze` 与 `create_job` 不同步** 的行为（仅改状态、不调用 `analyze_jd_text`），见 `job链路梳理.md` 第一节。 |

**原则**：尽量少改无关模块；鉴权与路由风格与 FastAPI 项目其余部分保持一致；表单创建路径避免依赖「必须 raw 文本才能落 active」的隐式规则。

---

## 2. 现状摘要（便于对齐）

- **API 入口目录**：`app/api/v1/endpoints/`（`/api/v1` 前缀）。
- **模型**：`UserModel`（`users`）、`JobModel`（`jobs`，含 `created_by_id`、`structured`、`raw_jd_text`）。
- **Job HTTP**：`jobs.py`；**尚无** `users`/auth 的 HTTP 模块（仅有 `app/schemas/user.py`）。
- **简历上传参考**：`resumes.py` + `parse_resume_document` — JD 文件上传可借鉴 multipart 与临时落盘方式。

---

## 3. 分阶段计划

### 阶段 A：鉴权与用户 API（User 链闭环）

1. 选定鉴权方案（建议 MVP：**JWT access token** 或 **Session + HttpOnly cookie**，二选一写死在配置里，避免双轨）。
2. 新增依赖：`bcrypt`（密码哈希）、`python-jose`（JWT）。实现上已用 `bcrypt` 直哈希，避免旧版 `passlib` 与新版 `bcrypt` 的兼容问题。
3. 实现 `UserRepository`（若不存在）：按 `username` 查、创建用户（`password_hash` 由 service 写入）。
4. 新增 `app/services/user_service.py`：`register`、`authenticate`、`get_by_id`。
5. 新增 `app/api/v1/endpoints/auth.py`（或 `users.py`）：  
   - `POST /auth/register`、`POST /auth/login`（或 `POST /users` + `POST /auth/token`）。  
   - 返回体含 `user` 摘要 + token（若用 JWT）。
6. 新增 FastAPI `Depends`：`get_current_user` / `get_current_user_optional`，从 `Authorization: Bearer` 或 cookie 解析。
7. 在 `app/api/v1/__init__.py` 注册新 router；`.env.example` 增加 `JWT_SECRET` / `ACCESS_TOKEN_EXPIRE_MINUTES` 等（若适用）。

### 阶段 B：Job 与当前 HR 绑定

1. **创建岗位**：`POST /jobs` 中若已登录，**忽略或覆盖** body 里的 `created_by_id`，统一写 `current_user.id`（管理员角色可后置扩展）。
2. **列表岗位**：`GET /jobs` 默认 `created_by_id=current_user.id`；可通过 query 显式放宽（仅 admin，可选）。
3. **读取/更新**：`GET/PATCH /jobs/{id}` 校验 `job.created_by_id == current_user.id`（或 public 只读策略需产品拍板）。
4. **Matching / tasks**：凡传入 `job_id` 的接口，建议同样校验 job 归属，防止跨租户读数据。

### 阶段 C：表单优先的 Job 创建契约

1. 与前端约定一份 **`structured` JSON schema**（以 `JobStructured` 为基线，字段命名统一 `snake_case`）。
2. **默认流程**：`POST /jobs`，body 含 `title` + `structured`，`raw_jd_text` 可选；`status` 建议直接 `active`（或 `draft` 再发布，需统一状态机文档）。
3. **Query `auto_analyze`**：当请求 **仅含表单**、无 `raw_jd_text` 时，`auto_analyze` 应为 no-op 或固定 `false`，避免误置 `analyzing`。
4. 可选：提供专用 schema `JobCreateFromForm`，在服务端将表单字段 **merge** 进 `structured`，减少前端重复。

### 阶段 D：`auto_analyze` 与创建路径一致化（缺陷修复）

**问题**：`create_job`/`update_job` 在 `auto_analyze` 下可能把 `status` 设为 `analyzing`，但 **不会** 调用 `analyze_jd_text`（与 `retry_analyze_job` 不一致）。

**可选实现（择一或组合）**：

| 方案 | 做法 |
|------|------|
| D1 | 在 `JobService.create_job`（及必要时 `update_job`）中，当存在 `raw_jd_text` 且无 `structured` 且 `auto_analyze` 为真时，**同步调用**与 `retry_analyze_job` 相同的分析逻辑（抽成 `_analyze_and_persist`），失败则 `failed` + `error_message`。 |
| D2 | 创建/更新 **不** 改状态为 `analyzing`；仅入队异步任务 `job_jd_analyze`，由 worker 写 `structured` 与状态（与现有 `job_profile_index_build` 任务模型对齐）。 |
| D3 | 文档与 API：`auto_analyze` 默认改为 `false`，并注明「结构化请调 `retry-analyze` 或上传后异步解析」（成本低但 UX 差）。 |

推荐：**D1（小流量同步）或 D2（与 task-worker 统一）**；实施时在 `docs/job链路梳理.md` 追加一节说明新行为。

### 阶段 E：JD 文件上传（次路径）

1. 新增 `POST /api/v1/jobs/upload`（或 `/jobs/from-file`）：`multipart/form-data`，字段：`file`、可选 `auto_analyze`。
2. 文本类：直接读 bytes 为文本或复用简历抽取中含 DOCX/PDF 的分支；图片：复用 **视觉/版面** 路径或限制首版只支持 PDF/TXT/DOCX。
3. 落库：创建 `JobModel`，`raw_jd_text=抽取全文`，`title` 由前端表单或自 filename / LLM 摘要生成（需约定）。
4. 调用与 **阶段 D** 一致的分析逻辑，再触发现有 `job_profile_index_build` 任务链。

### 阶段 F：测试与文档

1. **单测**：auth（注册登录、401）、job create（绑定 user、越权 patch/update）、form-only create + list filter。
2. **集成**：沿用 `tests/test_jobs_api.py` / `tests/test_hr_flow.py` 风格，增加带鉴权 header 的流程。
3. 更新 `docs/job链路梳理.md` 中「auto_analyze」与新建上传入口的描述。

---

## 4. 建议执行顺序（工单级）

1. **A1–A7**：用户注册登录 + `get_current_user`。  
2. **D**：确定并落地 `auto_analyze` 与创建路径一致（避免长期 `analyzing`）。  
3. **B**：Job 接口挂靠当前用户与越权校验。  
4. **C**：与前端锁定 `structured` 契约及默认 `status`。  
5. **E**：JD 上传接口（可按文件类型分迭代：先 txt/docx/pdf，后图片）。  
6. **F**：测试与文档同步。

---

## 5. 主要将触碰的路径（预估）

- 新增：`app/api/v1/endpoints/auth.py`（或等价）、`app/services/user_service.py`、`app/database/repository/user_repository.py`（若缺）、`app/core/security.py`（密码与 token，可选集中）。
- 修改：`app/api/v1/__init__.py`、`app/api/v1/endpoints/jobs.py`、`app/services/job_service.py`、可能 `app/api/v1/endpoints/matching.py`、`tasks.py`（归属校验）。
- 配置：`requirements.txt`、`.env.example`。
- 文档：`docs/job链路梳理.md`、本文件迭代记录。

---

## 6. 风险与依赖

- **跨接口一致性**：matching、hr-agent、task payload 均带 `job_id`，必须在服务层统一 **job 归属** 校验入口，避免遗漏。
- **密码与密钥**：生产环境必须配置强随机 `JWT_SECRET`；禁止把真实 `.env` 提交到仓库。
- **JD 图片**：与简历 OCR/视觉链路共用时要考虑费用与超时，适合放异步任务（阶段 D2 / E 结合）。

---

## 7. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-03-26 | 初稿：HR user 链、表单优先 Job、文件上传、`auto_analyze` 修复选项与执行顺序。 |
