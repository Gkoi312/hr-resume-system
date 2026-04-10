# Project Data — 实现结构与阅读地图

本文描述 **本仓库当前代码** 中的分层、数据表与主链路，便于从「入口 → 服务 → 存储 → 异步」逐层理解。

---

## 1. 本文档定位

| 读者目标 | 建议阅读顺序 |
|----------|----------------|
| 快速知道系统做什么 | 本节 + §2 + §5 |
| 跟代码读一条完整业务链 | §6（简历）+ §8（匹配）+ §7（任务） |
| 调语义 / 向量相关 | §9 |
| 配环境、接 OpenAPI | §11 + 根目录 `README.md`、`/.env.example` |

---

## 2. 本仓库是什么、不是什么

**是什么**

- 一个 **FastAPI 后端**，HTTP 前缀为 **`/api/v1`**（见 `app/main.py`）。
- 管理 **岗位（Job）**、**简历（Resume）**、**候选人主档（Candidate）**，并对「岗位 ↔ 候选人」做 **批量匹配**，结果落在 **`candidate_job_matches`**，带 **可 JSON 序列化的解释**（`MatchExplanation` 结构，见 `app/schemas/match.py`）。
- 使用 **PostgreSQL**（推荐 + `pgvector` 存向量）、**异步 SQLAlchemy + asyncpg**；可选 **独立 Worker 进程**消费 `tasks` 表。

---

## 3. 分层架构（实现视角）

```
HTTP  /api/v1/*
        │
        ▼
app/api/v1/endpoints/*.py     ← 参数校验、HTTP 语义、job_access（部分接口）
        │
        ▼
app/services/*                ← 业务编排：matching、resume、job、candidate、task、education_gate、semantic_chunk_matching
        │
        ├──────────────┬──────────────────┬─────────────────────────┐
        ▼              ▼                  ▼                         ▼
app/database/    app/rag/*           app/parsers/*              app/llm/*
repository/*       vector_store       resume / job 解析            embedding、chat_client
        │         chunk_profiles
        │         hybrid_retrieval
        ▼
PostgreSQL（及可选 pgvector 列类型）
```

**约定**

- **Repository**：`app/database/repository/` 封装表级 CRUD 与查询，由 Service 调用。
- **RAG 在本项目中的含义**：为匹配服务的 **多轴语义分** 提供 **向量块**（非对话式 Agent）。
- **LLM**：可选用于 **简历结构化抽取**、**JD 结构化抽取**；**匹配打分主路径**为规则门槛 + 向量相似度融合，不依赖大模型给最终分数。

---

## 4. 核心数据模型（表级）

定义见 `app/database/models.py`。

| 表名 | 作用 |
|------|------|
| `users` | 注册用户信息；`jobs.created_by_id` 可关联，用于 `job_access` 数据范围 |
| `jobs` | 岗位：`raw_jd_text`、`structured`（JSON）、显式列如 `min_years`、`education_requirement`、`status` |
| `candidates` | 候选人主档：姓名、联系方式、教育/工作/项目/技能等 JSON 与标量字段；**匹配以主档为业务锚点** |
| `resumes` | 简历文件元数据 + `parsed`（解析结果 JSON）+ `status`；多简历可挂同一 `candidate_id` |
| `candidate_job_matches` | 某 `job_id` + `candidate_id` 一条匹配：`overall_score`、`semantic_score` 等 + **`explanation`（JSON）**；`(job_id, candidate_id)` 唯一 |
| `tasks` | 异步任务：`task_type`、`payload`、`status`、`result_summary` 等 |
| `vector_profiles` | 向量索引：**每个 (entity_type, entity_id, profile_type) 一行**，存一块文本对应的向量及 meta |
| `rag_evaluations` | 通用 RAG 评测记录（与主业务弱耦合，可忽略初读） |
| `llm_usage_logs` | LLM 调用统计（无正文内容） |

**向量存储后端**：由环境变量 `VECTOR_STORE_BACKEND` 控制；`pgvector` 时向量列为 pgvector 类型，否则退化为 JSON 存浮点列表（见 `models.py` 顶部逻辑）。

---

## 5. HTTP API 一览（真实路由）

路由聚合在 `app/api/v1/__init__.py`，前缀均为 `/api/v1`。

| 前缀 | 模块文件 | 主要职责 |
|------|-----------|----------|
| `/auth` | `endpoints/auth.py` | 注册、登录、JWT |
| `/jobs` | `endpoints/jobs.py` | 岗位 CRUD、JD 上传/解析、触发岗位向量构建任务等 |
| `/resumes` | `endpoints/resumes.py` | 简历上传、解析、与候选人绑定 |
| `/candidates` | `endpoints/candidates.py` | 候选人主档 CRUD、人才库列表 |
| `/matching` | `endpoints/matching.py` | 同步/异步触发匹配、按岗位查询匹配列表等 |
| `/tasks` | `endpoints/tasks.py` | 创建部分异步任务、查询任务状态 |

**权限**：`app/api/job_access.py` 中 `ensure_job_access` — 若请求带已登录用户，则仅允许访问 **`created_by_id` 为自己** 的岗位（历史无 `created_by_id` 的岗位仍放行）。

---

## 6. 简历 → 候选人：同步与异步边界

**典型路径（概念）**

1. 上传简历 → 落 `resumes`、抽取文本 → **解析 pipeline**（`app/parsers/`：含 Paddle、可选 `RESUME_LLM_*` 文本/视觉 LLM）→ 写入 `resumes.parsed`。
2. **绑定候选人**：`candidate_profile_builder` 等逻辑将解析结果合并进 **`candidates`**（规则见 `resume_service`：补缺、不随意覆盖 HR 已编辑字段）。
3. 可 enqueue **`candidate_profile_index_build`**，由 Worker 根据候选人结构化文本 **切块并 embedding**，写入 **`vector_profiles`**（多种 `profile_type`，见 §9）。

**代码入口**：`app/services/resume_service.py`、`app/parsers/resume_parser/`、`app/tasks/resume_task.py`。

---

## 7. 异步任务（Worker）

进程入口：`python -m app.workers.task_worker`（`app/workers/task_worker.py`）。

**支持的 `task_type`（与代码一致）**

| task_type | 处理函数 | 用途 |
|-----------|-----------|------|
| `resume_upload` | `app/tasks/resume_task.py` | 异步完成上传/解析/绑定等重路径 |
| `matching_run` | `app/tasks/match_task.py` | 异步跑批匹配 |
| `job_profile_index_build` | `app/tasks/rag_index_task.py` | 岗位侧语义块向量写入 `vector_profiles` |
| `candidate_profile_index_build` | `app/tasks/rag_index_task.py` | 候选人侧语义块向量写入 `vector_profiles` |

Worker 通过 `task_repository.acquire_next_pending` 使用 **`FOR UPDATE SKIP LOCKED`**，可多实例并发消费。

---

## 8. 匹配流水线（`matching_service`）

实现与注释见 `app/services/matching_service.py`，与 README「匹配链路」一致，此处按 **执行顺序** 压缩为层次说明：

1. **加载候选人池**  
   - 若调用方传入 `candidate_ids` 则用之；否则从人才库 **`list(limit=…)`** 取一批（上限以代码为准，README 中写的是 5000 量级）。

2. **学历硬门槛（简历侧）**  
   - `education_resume_gate.filter_candidates_by_resume_education`：依据岗位 `education_requirement` 与 **`Resume.parsed` 中的学历信息** 过滤。  
   - **未通过者不参与后续语义计算、不写匹配行**。

3. **语义综合分（批量）**  
   - `semantic_chunk_matching.compute_semantic_scores_for_candidates_bulk`：读 **`vector_profiles`** 中 job/candidate 各 `profile_type` 的向量块，多轴融合为 **`semantic_score`**（并产生解释用片段、Delivery 对齐明细等）。  
   - 若候选人侧索引缺失，会得到相应 **`semantic_status`**（如未索引），解释里会提示需建索引。

4. **落库**  
   - 对每个通过门槛的候选人在 `candidate_job_matches` **upsert 一条**（唯一键 `job_id + candidate_id`）。  
   - **`overall_score` / `semantic_score`** 当前与语义主路径对齐；`skill_score` 等可为空（预留）。  
   - **`explanation`**：Pydantic `MatchExplanation` 转 dict，含硬条件摘要、亮点/风险、`semantic_evidence`、`delivery_alignments`（混合检索开启时）等。

5. **列表排序**  
   - API 侧按分数 **降序** 返回；不是只保留第一名。

**设计原则**：硬门槛可审计；向量用于排序与解释增强，避免单一黑盒模型直接决定录用。

---

## 9. 语义子系统：块、向量、Delivery 混合检索

**块定义与文本来源**  
- `app/rag/chunk_profiles.py`：从 `JobModel` / `CandidateModel`（及 structured 字段）构造多块文本，并生成稳定的 **`profile_type`** 名称（如 `skill`、`resp` / `resp_<n>`、`proj_<n>`、`work_<n>`、`role` 等，长度受限）。

**写入**  
- `app/tasks/rag_index_task.py` 调用 embedding（`app/llm/embedding.py`），经 `app/rag/vector_store.py` 的 **`sync_entity_vector_chunks`** 写入 **`vector_profiles`**。

**匹配时融合**（`semantic_chunk_matching.py`）  

- **轴权重（常量）**：技能、**Delivery（职责↔经历/项目）**、角色 — 代码内 `WEIGHT_SEMANTIC_*` 定义。  
- **Delivery**：默认 **`SEMANTIC_DELIVERY_HYBRID=1`** 时，对职责块与经历块做 **余弦排名 + BM25 排名 → RRF 融合** 选对齐对；关闭时退回纯余弦最大对思路。  
- **RRF 平滑常数**：`SEMANTIC_RRF_K`（默认 60）。BM25 侧中文分词见 `app/rag/hybrid_retrieval.py` 与 **`app/rag/jieba_userdict.txt`**。

**解释输出**  
- `MatchExplanation.delivery_alignments`：职责片段与候选人片段对齐明细（余弦、BM25、RRF、排名、`shared_terms` 等），供前端表格/侧栏展示。

---

## 10. LLM 与 `chat_client`（边界说明）

| 用途 | 配置前缀 | 代码入口（示例） |
|------|-----------|------------------|
| 简历 layer1 结构化 JSON | `RESUME_LLM_*` / `RESUME_VLLM_*` | `app/parsers/resume_parser/resume_llm_layer1/extract.py` |
| JD 转 `JobStructured` | `JOB_LLM_*` | `app/parsers/job_parser/extract.py` |
| 通用 HTTP Chat 客户端 | 无参默认 **`RESUME_LLM_*`** | `app/llm/chat_client.py` 中 `ChatLLMClient` |

解析是否启用由各自 `*_ENABLED` 环境变量控制；**匹配服务不调用 Chat LLM 打分**。

---

## 11. 配置与环境变量（从哪读）

- **数据库**：`DATABASE_URL`  
- **向量后端与维度**：`VECTOR_STORE_BACKEND`、`EMBEDDING_*`、`EMBEDDING_DIM`（须与模型与列维度一致）  
- **语义 Delivery**：`SEMANTIC_DELIVERY_HYBRID`、`SEMANTIC_RRF_K`  
- **Worker**：`TASK_WORKER_POLL_INTERVAL`、`TASK_WORKER_MAX_CONCURRENCY`、`TASK_WORKER_ID`  

完整示例与说明见 **根目录 `.env.example`** 与 **`README.md`**。

---

## 12. 建议的代码阅读路径（按小时粒度）

1. **`app/main.py`** + **`app/api/v1/__init__.py`**：路由全貌。  
2. **`app/services/matching_service.py`**：匹配主流程与 `MatchExplanation` 组装。  
3. **`app/services/semantic_chunk_matching.py`** + **`app/rag/chunk_profiles.py`**：语义分与块类型。  
4. **`app/rag/vector_store.py`** + **`app/tasks/rag_index_task.py`**：向量如何落库。  
5. **`app/services/resume_service.py`** + **`app/parsers/resume_parser/`**：简历进主档。  
6. **`app/workers/task_worker.py`**：异步任务分发列表。

---

## 13. 与其它文档的关系

- **`README.md`**：面向使用与部署的精简说明。  
- **`docs/`** 下各链路文档：可能包含历史方案或表名（如旧文档中的 `resume_chunk_embeddings`）；**若与 §4、§9 不一致，以本文与 `app/database/models.py` 为准**。  
- **`json_design/`**：简历/解析 JSON 与 LLM 提示相关设计备忘。

---
