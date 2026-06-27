# Job / Candidate 分块向量化方案与执行计划

> 范围：仅描述 **embedding 文本构造、存储契约与落地步骤**；`industry_preference` 不参与向量化；**学历与最低年限** 仅用于硬性过滤，不进向量。  
> 存储仍沿用 `vector_profiles` 表：`entity_type` ∈ `job` | `candidate`，用 **`profile_type`** 区分语义块（单表多行，每行一个向量）。

---

## 1. 存储契约（`vector_profiles`）

| 字段 | 约定 |
|------|------|
| `entity_type` | `job` 或 `candidate` |
| `entity_id` | `jobs.id` / `candidates.id` |
| `profile_type` | 短字符串区分块，见下文枚举（须 ≤32 字符，兼容当前 ORM 定义） |
| `vector` | 与全局 `EMBEDDING_DIM` / 模型一致 |
| `content_hash` | 仅对该块 **canonical 拼接文本** 做 SHA256，变更则重算 embedding |
| `meta` | 建议含 `semantic_text`（或截断预览）、`block` 与可选 `index`（列表类第几条） |
| `status` | `available` / `embedding_failed`；失败时零向量占位 + `error_message`（与现实现一致） |
| `embedding_model` | 当前 embedding 模型名，便于切换模型后全量重算 |

**唯一约束**：现表为 `(entity_type, entity_id, profile_type)` 唯一。列表类多块使用带序号后缀的 `profile_type`（如 `proj_0`、`work_1`），或使用固定前缀 + `meta.index`。

---

## 2. Job 侧：分块定义

**不进向量的字段**（仅 Gate / 结构化规则）：

- `structured.min_years`、Job 表 `min_years` 列（若后续同步）
- `structured.education_requirement`、`education_requirement` 列

**不参与本方案**（按产品决定不做 embedding）：

- `industry_preference`

### 2.1 `skill`

- **来源**：`structured.required_skills`、`structured.preferred_skills`（可带前缀区分「必备」「加分」）。
- **空块**：若无任何技能，可跳过写入或写 `status=skipped` 类约定（实现时二选一，建议跳过不插行，匹配侧按「无技能向量」处理）。

### 2.2 `role`

- **来源**：`Job.title` + `structured.job_title` + `structured.job_summary`；非空字段换行拼接。
- **可选补充**：`structured.keywords` 若有，附在末尾一句（不单独成块）。

### 2.3 `resp`

- **来源**：`structured.responsibilities`（`list[str]`）合并为一段；条目之间用换行或 ` | ` 分隔。
- **过长**：先 MVP 整条拼接；超长再按条拆 `resp_0`、`resp_1`… 或截断（执行阶段实现时定长度阈值）。

### 2.4 `raw_jd_text`

- **默认**：不作为独立必选块；仅当 `responsibilities` 与 `job_summary`（及 `job_title`）均不足以构成 `role`/`resp` 时，可增加兜底块 `jd_raw`（实现可选，避免与 `resp`/`role` 重复时可只在 meta 标注 `fallback`）。

---

## 3. Candidate 侧：分块定义

**不进向量的字段**（仅 Gate）：

- `education`（JSON 列表）：学历门槛、排序在匹配 Gate / 结构化分中处理

### 3.1 `skill`

- **来源**：`candidates.skills`（列表）；拼接为一句「技能：…」或与 Job `skill` 块风格一致的模板。

### 3.2 `proj_{i}`

- **来源**：`candidates.projects` 每一项：`name`、`role`、`description` 拼成一段；`i` 为从 0 递增的序号，与列表顺序一致。
- **空列表**：不产生行。

### 3.3 `work_{i}`

- **来源**：`candidates.work_experience` 每一项：`company`、`title`/`position`、`description` 拼成一段；序号规则同 `proj_*`。

### 3.4 `cand_role`

- **来源**（与产品约定一致）：
  - `direction_tags`；
  - 项目条数、工作/实习条数（摘要数字）；
  - 各 `project.name`、各 `work` 的公司与职位名称（简短罗列，避免整段 description 再次挤爆单块）。
- **目的**：与 Job 的 `role`+`resp` 在「方向 / 职务形态」上对齐全路语义；描述性细节由 `proj_*` / `work_*` 承担。

---

## 4. 与匹配三路的对应关系（供后续 Match 实现引用）

| 语义路 | Job 块 | Candidate 块 |
|--------|--------|----------------|
| 技能语义 | `skill` | `skill` |
| 项目 / 经历语义 | `resp`（及可选 `jd_raw`） | `proj_*`、`work_*` |
| 方向 / 角色语义 | `role` | `cand_role` |

具体融合权重与 Top-K、rerank 由 Match 链设计文档另行定稿；本文件只保证 **块边界与库表一致**。

---

## 5. 执行计划

### Phase 1：文本构建与哈希（纯函数、可单测）

1. 新增或扩展模块（建议路径示例）：`app/rag/chunk_builder_job.py`、`app/rag/chunk_builder_candidate.py`（或合并为 `chunk_profiles.py` 内两族函数）。
2. 实现 `build_job_chunks(job: JobModel) -> list[ChunkSpec]`、`build_candidate_chunks(candidate: CandidateModel) -> list[ChunkSpec]`，`ChunkSpec` 含 `profile_type`、`text`、`content_hash` 元数据。
3. 单元测试：覆盖空字段、仅 partial structured、多项目多工作经历序号稳定。

### Phase 2：写入 `vector_profiles`

1. 扩展 `vector_store`：按 `entity_type`、`entity_id`、`profile_type` **upsert**；支持一批块同一事务或逐块提交（与现 `get_session_context` 一致）。
2. 删除/失效策略：Job 或 Candidate 更新后，**本次未出现的 `profile_type` 旧行**应删除或标记废弃，避免残留错误块（实现二选一：先删后插，或 `sync_chunks` 全量对比）。
3. 复用现有异步任务类型或拆分：
   - 保留 `job_profile_index_build` / `candidate_profile_index_build`，内部改为 **多块 embed + 多行 upsert**；
   - 或新增任务类型 `job_chunks_index_build` / `candidate_chunks_index_build` 并切换 worker（择一，避免重复触发）。

### Phase 3：索引与模型

1. 环境：`EMBEDDING_PROVIDER`（如 bge / ollama）、`EMBEDDING_DIM`、`VECTOR_STORE_BACKEND=pgvector` 时确保维一致。
2. 批量 `embed_texts`：每 Job/Candidate 一次请求传多块文本，减少 RTT。
3. pgvector：对 `(entity_type, entity_id)` 查询频繁的可在应用侧过滤；若后续做 ANN，再为 chunk 表或扩列建 HNSW（非本阶段必须）。

### Phase 4：兼容与迁移

1. 旧的 `profile_type=general`：迁移策略 ——— 首次发布可多写 `general`（全文兜底）一段时间后下线，或直接停止写入 `general`、匹配侧仅读新块。
2. 数据回填脚本：对已有 job/candidate 批量投递索引任务或离线跑一轮 `ChunkSpec` + upsert。

### Phase 5：Match 链

1. 新匹配逻辑按上表三路分别取块算相似度（或 max-over-chunks），不再依赖单一 `general` 向量。
2. Gate 使用 `min_years`、`education_requirement`（Job）与 `education`（Candidate），与本文 embedding 范围解耦。

---

## 6. 文档与代码索引

- Job 结构化字段：`app/schemas/job.py`（`JobStructured`）
- Candidate ORM：`app/database/models.py`（`CandidateModel`）
- 现单向量任务：`app/tasks/rag_index_task.py`、`app/rag/profile_builder.py`
- 现向量存储：`app/rag/vector_store.py`、`VectorProfileModel`

---

## 7. 修订记录

- 初版：按产品约定去掉 `industry_preference` embedding；Job 三块 `skill` / `role` / `resp`；Candidate 四族 `skill` / `proj_*` / `work_*` / `cand_role`；学历与年限仅 Gate。
