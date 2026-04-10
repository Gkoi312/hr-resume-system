# HR Resume System

面向 **HR 初筛实习生/校招简历** 的后端服务：管理岗位（JD）、简历与候选人主档，在 **学历等硬门槛** 之上用 **规则 + 语义向量** 对「岗位 ↔ 候选人」批量打分，生成 **可解释的匹配记录**；支持异步任务与向量索引。

---

## 目标用户与解决的问题

- **谁在用**：招实习生或应届的 HR / 用人助理，同一岗位可能收到大量投递。  
- **系统做什么**：把简历变成结构化候选人画像，按岗位聚合并 **排序**，并给出 **理由摘要 + 语义证据片段**，减轻逐份阅读与内部沟通成本。  
- **系统不做什么**：不做最终录用决策；不替代人工复核敏感门槛；大模型（若开启）仅作 **叙述/润色**，核心分数与结构化字段以规则与向量为先。

---

## 核心能力（当前实现）

| 能力 | 说明 |
|------|------|
| **岗位 Job** | 保存 JD 原文与 `structured` JSON（必备/优先技能、年限、学历、行业等）及显式列；可触发岗位侧向量索引。 |
| **简历 Resume / 候选人 Candidate** | 上传 PDF/DOCX 等 → 抽取文本 → 解析 pipeline → 回填 **候选人主档**。业务上的匹配与检索以 **Candidate** 为主；解析可补缺，不随意覆盖 HR 已编辑字段。 |
| **匹配 Matching** | 对指定 `job_id` 与 **候选人池**（显式 ID 列表或名单库截断列表）逐人：先 **简历学历硬过滤**，再算 **语义综合分**，每人写入一条 **`candidate_job_matches`**；返回列表按分数 **降序**，不是只保留「第一名」。 |
| **语义 / 「类 RAG」** | 岗位与候选人的多块语义文本经 **Embedding** 后写入 **`vector_profiles`**。匹配时在应用内做多轴聚合（技能 / **交付 Delivery** / 角色）。**Delivery 轴**对每条职责与每段项目/实习经历：用 **余弦排名 + BM25 排名做 RRF 融合** 选对齐对，再用该对的余弦映射分数档；`delivery_detail.delivery_alignments` 记录 **职责块 ↔ 经历块** 及 RRF/BM25/排名（可关混合检索见环境变量）。 |
| **异步任务 Task** | 简历上传批处理、匹配跑批、RAG 索引等写入 `tasks` 表，由独立 **Worker** 消费，避免拖慢 HTTP。 |
| **认证 Auth** | 用户注册/登录；部分岗位接口可按当前用户做 **数据范围** 控制（见 `job_access`）。 |

---

## 匹配链路

1. **候选人池**：`run_matching(job_id, candidate_ids=...)` 若未传 ID，则使用名单库 `list(limit=5000)`（上限以代码为准）。  
2. **学历门槛**：`filter_candidates_by_resume_education` — 未过门槛者 **不参与** 语义打分、**不写入** 匹配表。  
3. **语义分**：`compute_semantic_scores_for_candidates_bulk` — 在 `semantic_chunk_matching` 中融合技能 / Delivery / 角色子分。Delivery 默认 **RRF（向量 + BM25）** 做职责–经历对齐；`SEMANTIC_DELIVERY_HYBRID=0` 时退回纯 max 余弦。  
4. **落库**：对每个通过门槛的候选人 `match_repository.create` 一条记录；`overall_score` / `semantic_score` 当前与语义综合分对齐；`skill_score`、`experience_score` 等字段可为空（预留多维扩展）。  
5. **解释**：`MatchExplanation` 含硬条件摘要、`summary_for_hr`（规则拼接）、`semantic_evidence`；混合检索开启时另有 **`delivery_alignments`**（每条职责 ↔ 最佳经历块：片段、cosine/BM25/RRF/排名、**重合词 `shared_terms`**），供 HR 可视化展开，与简历大表互补。

设计原则：**硬门槛尽量规则化、可审计；语义用于排序与解释增强**，避免单一黑盒大模型直接决定录取与否。

---

## 架构概览

```
客户端 ──HTTP──► FastAPI（app/main.py，前缀 /api/v1）
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   services/*   repository/*   rag/* + llm/*
        │           │           │
        └───────────┴───────────┘
                    ▼
             PostgreSQL（asyncpg）
             · jobs / resumes / candidates / candidate_job_matches
             · tasks · vector_profiles · users …

异步：python -m app.workers.task_worker
     （SKIP LOCKED 取任务，可多实例）
```

- **分层**：`endpoints` → `services` → `repository` / `vector_store`。  
- **向量后端**：`app/rag/vector_store.py` 默认将向量落在 **PostgreSQL**；也可用内存后端做本地调试（见环境变量说明）。

---

## 环境要求

- Python **3.10+**  
- **PostgreSQL**；`DATABASE_URL` 建议 `postgresql+asyncpg://...`（脚本或文档可能对 `postgresql://` 做转换）

---

## 快速开始

```bash
pip install -r requirements.txt
```

复制并填写环境变量（示例见 `.env.example` ）。

**启动 API：**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

启动时会 `init_db()`（`create_all`）建表。交互文档：<http://127.0.0.1:8000/docs>

**启动 Worker（异步任务与索引）：**

```bash
python -m app.workers.task_worker
```

可调：`TASK_WORKER_POLL_INTERVAL`、`TASK_WORKER_MAX_CONCURRENCY`、`TASK_WORKER_ID`。

---

## API 一览（`/api/v1`）

| 前缀 | 说明 |
|------|------|
| `/auth` | 注册、登录、令牌 |
| `/jobs` | 岗位 CRUD、上传/解析 JD |
| `/resumes` | 简历上传、解析、与候选人绑定等 |
| `/candidates` | 候选人主档、人才库查询 |
| `/matching` | 同步跑匹配、按岗位列匹配结果等 |
| `/tasks` | 异步任务创建与状态 |

---

## 重要环境变量（摘录）

- **数据库**：`DATABASE_URL`  
- **Embedding**：见 `app/llm/embedding.py` 与 `.env`  
- **向量后端**：见 `vector_store` 与相关 `VECTOR_*` 变量  
- **Delivery 混合检索**：`SEMANTIC_DELIVERY_HYBRID` 默认开启（设为 `0`/`false`/`off` 则纯余弦）；`SEMANTIC_RRF_K` 为 RRF 平滑常数（默认 `60`）。BM25 中文分词使用 **jieba**；自定义词表 **`app/rag/jieba_userdict.txt`** 在进程内 **导入时加载一次**（如「后端」「校招」等），可按业务增删行。

---

## 仓库结构（核心）

```
hr_resume_system/
├── app/
│   ├── main.py
│   ├── api/v1/endpoints/     # auth, jobs, resumes, candidates, matching, tasks
│   ├── services/             # matching_service, resume_service, …
│   ├── database/             # models, session, repository
│   ├── tasks/                # resume / matching / rag_index 处理器
│   ├── workers/task_worker.py
│   ├── rag/                  # chunk 构建、vector_store、hybrid_retrieval（BM25+RRF）
│   ├── parsers/              # JD / 简历解析
│   └── llm/                  # embedding.py, chat_client.py
├── scripts/                  # 验证与工具脚本
├── docs/                     # 设计备忘
├── project_data.md           # 系统设计与数据流（若需深入可读）
├── requirements.txt
└── README.md
```

---

## 验证脚本

```bash
python scripts/verify_rag_v11_semantic.py   # 语义匹配与证据
```

更多脚本说明见 `scripts/README.md`（若存在）。

---

## 其它目录说明

仓库中若包含 **`reference_project/`**、第三方样例工程等，**与本项目运行方式无关**，部署与阅读时可忽略。

---

## 技术栈

- FastAPI、Uvicorn、Pydantic v2  
- SQLAlchemy 2.x asyncio、asyncpg、PostgreSQL  
- 简历/JD 解析：python-docx、PyMuPDF 等（OCR/视觉路径视环境额外安装）  
- 向量：sentence-transformers / HTTP Embedding + **pgvector**（`requirements.txt` 为准）

---

## License

以仓库内许可证文件为准；若未附带，请注明使用范围。
