# pgvector + BGE 的 Job-Candidate Matching 向量化执行 Plan

> 目标：job 只与 candidate 做语义匹配（resume 只用于构造 candidate 画像数据，不再作为可检索证据进行 embedding / 检索）。

## 0. 现状与约束（先对齐范围）
- 已完成：`resumes` / `candidates` 的结构化数据落库（PostgreSQL）。
- 需要做：把语义匹配的 VectorStore 从当前实现替换为 **pgvector**；把 embedding 从 debug-hash 替换为 **BGE（建议 BAAI/bge-small-zh-v1.5）**。
- 需要重新约束：matching 只依赖
  - `job` 向量（job profile embedding）
  - `candidate` 向量（candidate profile embedding）
  - 不依赖 `resume`/evidence chunks 的 embedding 检索。

## 1. 向量库选择与扩展（pgvector）
1. 安装依赖（Python 侧与 ORM 侧按你的实现方式选一种即可）：
   - 推荐：`pgvector`（用于 SQLAlchemy 的 vector 类型映射与/或常用操作）
2. Postgres 开启扩展：
   - 在目标数据库执行：`CREATE EXTENSION IF NOT EXISTS vector;`
3. 表与索引：
   - `vector_profiles`：存 `job` / `candidate` 的 general vector
   - （可选但不建议 MVP 阶段继续保留）`resume_chunk_embeddings`：如果不需要 evidence，MVP 可以先不建或不写
   - 为 `vector` 列建 ANN 索引（按 pgvector 版本选择 `hnsw` 或 `ivfflat`）

## 2. embedding 模型（BGE）与 embedding 文本块设计
### 2.1 模型选择
- embedding：`BAAI/bge-small-zh-v1.5`
- 输出向量归一化：必须启用（等价于 normalize embeddings），以便 cosine 相似度映射稳定。
  - 参考项目（CookHero）关键点：`normalize_embeddings=True`

### 2.2 Job embedding 输入块（只用岗位需求）
- 构造一个稳定字段顺序的拼接文本（MVP 即可）：
  - `title`（如有）
  - `required_skills`
  - `preferred_skills`
  - `min_years`
  - `education_requirement`
  - `industry_preference`
  - `raw_jd_text`（作为兜底）
- 说明：job structured 字段优先，raw 兜底，避免结构缺失导致语义信息过少。

### 2.3 Candidate embedding 输入块（重点：skills / education / work / projects）
- 重要调整：**不使用 summary**（summary 的语义来自候选人的自我评价，主观且噪音高）。
- 仅拼接这些字段（字段顺序固定，且 topN 限制以控制长度）：
  - `skills`
  - `education`（如果想更省 token：也可只放 `highest_education`；但更全面可放 education 列表）
  - `work_experience`：取最近/最相关 top3
    - 每条拼：`company + title/position + description`
  - `projects`：取 top2
    - 每条拼：`project_name + role + description`
- 去重规则：
  - skills 去重
  - projects/work 保持原抽取 index / 顺序（不做文本级模糊去重，避免成本飙升）

### 2.4 Candidate 多份简历如何落到 embedding（当前的 resume 定位）
- resume 只用于更新/补齐 candidate 结构化字段。
- candidate embedding 输入来自 **candidate 当前画像字段快照**。
- 因此：当同一 candidate 上传多份 resume 时，只有当新 resume 更新了“原本为空”的字段，candidate 文本才会变，embedding 才需要重建。

## 3. JSON / 元信息（meta）设计：让向量重建可控
建议把 embedding 的“输入文本片段、content_hash、参与字段”写入 `meta`（JSONB），便于调试与回滚。

`vector_profiles.meta` 建议结构示例：
```json
{
  "embedding_text": "【拼接后的 job/candidate 文本】",
  "embedding_fields": ["skills", "education", "work_experience", "projects"],
  "text_truncation_rules": {"work_desc_max_chars": 600, "project_desc_max_chars": 600},
  "embedding_version": "bge-small-zh-v1.5-normalized-v1",
  "rebuild_reason": "candidate updated fields"
}
```

`content_hash`：
- 对 `embedding_text` 做 sha256，作为是否需要重建的判定依据。

## 4. 向量化任务与 worker 调整（MVP：仅 job/candidate general 向量）
### 4.1 保留的任务类型
- `job_profile_index_build`
- `candidate_profile_index_build`

### 4.2 不再使用的任务类型（MVP 可禁用）
- `resume_index_build`
- `resume_chunk_embeddings` 写入相关逻辑

### 4.3 触发策略（在结构化数据更新后触发）
- job 创建/更新完成后：enqueue `job_profile_index_build`
- bind_candidate_from_resume 更新 candidate 字段后：enqueue `candidate_profile_index_build`
  - 触发发生点建议继续放在 `ResumeService.bind_candidate_from_resume()` 末尾

## 5. pgvector 写入（upsert）与读取（检索/相似度）
### 5.1 写入
- `upsert`：根据 `(entity_type, entity_id, profile_type)` 唯一键覆盖向量与 meta
- 写入字段：
  - `vector`
  - `meta`
  - `content_hash`
  - `embedding_model`
  - `status` / `error_message`

### 5.2 读取
- matching 阶段仅需要：
  - job 向量（拿到 job_vec）
  - candidate 向量相似度 topK
- 方式二选一：
  1. SQL 计算相似度并 topK（推荐）：性能更好，逻辑统一
  2. 先取候选人向量到 Python 再算 cosine（实现快，但可能慢）

> 注意：你的语义分区间（0.84/0.78/…）需要与“BGE cosine 相似度分布”匹配。
> 如果 pgvector 返回的是 cosine 距离，需要转换为相似度，再映射到 0~100 分。

## 6. matching_service 的改造（只做 job-candidate semantic）
1. 删除（或绕过）现有 evidence 依赖：
   - 不再调用 `search_resume_chunks_for_candidates`
   - 不再读取 `ResumeChunkEmbeddingModel`
   - `semantic_evidence` 固定为空
2. `_compute_semantic_scores_for_candidates()`：
   - 输入 job：从 pgvector 读取 job_vec（或 SQL 直接完成匹配）
   - 输出 candidate 的 `semantic_score` 和 `semantic_status`
3. JSON 字段保持兼容：
   - `semantic_status`：来自 `vector_profiles.status`
   - `semantic_evidence`：返回空列表（或 None，但前端要能接受）

## 7. 与参考项目的对应关系（CookHero-main）
- CookHero 使用 Milvus：
  - embedding：`BAAI/bge-small-zh-v1.5`
  - normalize：`normalize_embeddings=True`
  - 切块+索引+检索：有统一的 vector store 工厂与 embeddings 工厂
- 你们本项目已具备类似“任务构建向量 + matching 读取向量”的骨架：
  - `rag_index_task.py`：向量入库
  - `matching_service.py`：向量读取与打分
  - `vector_store.py`：向量后端抽象
- 本 Plan 的落点：
  - 复用你们的抽象层，只替换 `vector_store` 后端实现为 pgvector，并更新 embedding client 为 BGE。

## 8. 验收与测试建议
1. 初始化数据库后执行 worker，确保向量表有数据：
   - `vector_profiles` 至少有 job/candidate 对应向量行
2. 匹配结果语义分不为 0（相似度可用）：
   - 在匹配接口或脚本中打印 `semantic_status` 与 `raw_similarity`
3. token/截断策略导致的稳定性：
   - 同一候选人上传多份简历后，candidate content_hash 应该随“补齐字段”变化而变化

## 9. 实施顺序（推荐）
1. 先实现 BGE embedding（把 debug-hash 替换为真实 embedding，保证 normalize 与维度一致）
2. 再改 pgvector schema + vector_store 写入读取（MVP 只 job/candidate general）
3. 最后改 matching_service（移除 evidence 依赖，只使用 job-candidate 相似度）
4. 再处理 worker 任务触发与禁用 resume chunks 相关流程

