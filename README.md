# HR Resume System

[![Run Tests](https://github.com/gaojiaxin/hr_resume_system/actions/workflows/test.yml/badge.svg)](https://github.com/gaojiaxin/hr_resume_system/actions/workflows/test.yml)

面向 **HR 初筛实习生/校招简历** 的后端服务：管理岗位（JD）、简历与候选人主档，在 **学历硬门槛** 之上按 **技能匹配度 × 经历可信度** 对「岗位 ↔ 候选人」批量打分，生成 **可解释的匹配记录**（含技能命中/缺失列表、职责-经历对齐证据片段）；支持异步任务与混合检索。

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
| **匹配 Matching** | 对指定 `job_id` 与 **候选人池**（显式 ID 列表或名单库截断列表）逐人：先 **简历学历硬过滤**，再算 **技能规则分** 与 **经历质量分**，按 `skill × quality_factor` 乘法公式得综合分；每人写入一条 **`candidate_job_matches`**，返回列表按分数 **降序**。 |
| **语义 / 混合检索** | 岗位与候选人的多块语义文本经 **Embedding** 后写入 **`vector_profiles`**。BM25（纯 Python Okapi + jieba 分词 + 自定义词表）与向量余弦通过 **RRF 融合**，做职责–经历精确对齐，产出 `delivery_alignments` 证据表（含 shared_terms 重合词）。**不参与最终综合分排名**——仅用于可解释性展示，HR 可审计匹配依据。 |
| **异步任务 Task** | 简历上传批处理、匹配跑批、RAG 索引等写入 `tasks` 表，由独立 **Worker** 消费，避免拖慢 HTTP。 |
| **认证 Auth** | 用户注册/登录；部分岗位接口可按当前用户做 **数据范围** 控制（见 `job_access`）。 |

---

## 匹配链路

1. **候选人池**：`run_matching(job_id, candidate_ids=...)` 若未传 ID，则使用名单库 `list(limit=5000)`（上限以代码为准）。  
2. **学历门槛**：`filter_candidates_by_resume_education` — 从简历结构化数据提取最高学位档位（博士>硕士>本科>大专），与岗位最低要求比较。未过门槛者 **不参与** 后续打分、**不写入** 匹配表；简历未解析者标记 `unknown` 准许通过，不丢弃。  
3. **技能分**：岗位必备+优先技能与候选人技能集归一化后取交集，`100 × matched / required`。别名映射保证 `k8s` 命中 `kubernetes`，但框架与语言保持独立（`django` ≠ `python`）。  
4. **经历质量分**：纯规则四维度评估（内容充实度 35% + 具体性 30% + 公司认可度 20% + 经历广度 15%），可替代 LLM 路径（`MATCH_LLM_ENABLED=0`），零 API 成本、毫秒级延迟。  
5. **综合分**：`overall = skill_score × (0.6 + 0.4 × quality_score / 100)`。技能分决定排序上限，质量分作为可信度折扣系数（范围 0.6–1.0）。公式由消融实验验证，详见 [评测结果](#评测结果)。  
6. **解释**：`MatchExplanation` 含硬条件摘要、`summary_for_hr`、技能命中/缺失列表；混合检索（BM25 + 向量余弦 + RRF 融合）产出 **`delivery_alignments`**（每条岗位职责 ↔ 最佳候选人经历块：片段、cosine/BM25/RRF/排名、**重合词 `shared_terms`**），供 HR 可视化审计。  
7. **落库**：对每个通过门槛的候选人 `match_repository.create` 一条记录，含 `overall_score`、`skill_score`、`quality_score` 及各维度解释字段。

设计原则：**硬门槛规则化、可审计；技能关键词决定排名上限，质量分调节可信度；避免单一黑盒大模型直接决定录取与否**。

---

## 评测结果

匹配公式经 **消融实验** 验证——构造 15 岗位 × 40 候选人评测集（600 标注对），固定原始分后换不同权重配置重新排名，对比 ground truth 的 NDCG/Precision/Recall。

| 配置 | NDCG@5 | 说明 |
|------|--------|------|
| **Skill × Quality（当前公式）** | **1.0000** | 15/15 岗位均为最优 |
| SkillOnly（纯关键词） | 0.9948 | 14/15 最优 |
| Baseline（旧版三维加权） | 0.9747 | 被全面超越 |
| NoSem→Skill（去语义分） | 0.9828 | 去掉语义分反而更好 |
| SemanticOnly | 0.8545 | 有信号但漏人 |
| QualityOnly | 0.2347 | 基本随机（岗位无关） |

**结论**：语义向量分作为加法维度引入噪声，已从最终公式移除。质量分是岗位无关的，不能做加法维度，但作为乘法折扣系数有效——技能分决定排序上限，质量分调节可信度。

**排序质量**（当前公式，15 jobs × 40 candidates）：

| K | Precision | Recall | NDCG |
|---|-----------|--------|------|
| @1 | **1.0000** | 0.2814 | **1.0000** |
| @3 | 0.9111 | 0.6441 | 0.9740 |
| @5 | 0.7867 | 0.8213 | 0.9869 |
| @10 | 0.5200 | 0.9852 | 0.9852 |

- **MRR = 1.0000**：所有 15 个岗位最高相关候选人均排第一位
- **单次匹配延迟**：~125ms（40 候选人 / 单岗位）

**测试**：17 个测试文件、276 条用例、100% 通过。核心打分函数均覆盖纯函数单测（不走 DB、不调 LLM）。

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

## 🐳 Docker 一键启动

```bash
git clone <repo-url> && cd hr_resume_system
docker-compose up -d
```

启动后：
| 服务 | 地址 |
|------|------|
| API 文档 (Swagger) | http://localhost:8000/docs |
| Demo 前端 | http://localhost:8501 |
| 数据库 (PostgreSQL+pgvector) | localhost:5432 |

Docker Compose 自动启动 5 个服务：PostgreSQL + API + Worker + Seed（灌入 3 岗位 + 10 候选人）+ Demo 前端。

Demo 环境**零外部依赖**：embedding 用 debug 模式，无需 LLM API Key，开箱即用。

---

## 🖥️ Demo 前端

![Demo Screenshot](docs/screenshots/demo_overview.png)

Demo 提供 4 个页面：
- **岗位管理** — 列表查看 / 手动创建 / 上传 JD 文件自动解析
- **简历 & 候选人** — 批量上传简历（PDF/DOCX/图片），查看候选人画像
- **匹配分析** — 选岗位 → 学历过滤 → 多轴打分 → 可解释排序结果
- 每条匹配可展开查看：硬门槛 / 技能对比 / 语义证据 / Delivery 对齐表 / LLM 质量评估 / 分数拆解

```bash
# 单独启动 Demo（需要后端已在 8000 端口运行）
cd demo
pip install streamlit pandas requests
streamlit run app.py
```

> 截图请补充至 `docs/screenshots/` 目录。

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
