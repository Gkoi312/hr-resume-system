# Job JSON 规范（前端填表/后端/上传抽取统一契约）计划与执行

目标：先把 **Job JSON（标准化结构）**定义清楚，这样前端填表才能按元素设计；当用户上传 `image/pdf/txt/word/doc` 的 JD 时，后端抽取也能把结果格式化回同一份 JSON。

本计划会对齐当前后端已有的数据契约：
- `app/schemas/job.py:JobStructured`（字段名、类型）
- `app/services/job_service.py`（什么时候会解析 `raw_jd_text` 生成 `structured`）
- `app/rag/profile_builder.py:build_job_semantic_text`（用于 job 向量化的字段）
- 匹配/Agent（后续使用这些字段做评分与解释）

---

## 1. 标准 Job JSON（Canonical）

建议统一以 `structured` 作为标准化核心 JSON（前端填表的输出、上传抽取的输出都最终落到这里）。

### 1.1 `JobStructured` 字段定义（对齐代码）

`app/schemas/job.py` 中已经定义了 `JobStructured`，字段如下（类型与可选性）：

```json
{
  "job_title": null,
  "required_skills": null,
  "preferred_skills": null,
  "min_years": null,
  "education_requirement": null,
  "industry_preference": null,
  "keywords": null,
  "job_summary": null
}
```

其中：
- `required_skills`: `list[str] | null`
- `preferred_skills`: `list[str] | null`
- `min_years`: `int | null`
- `education_requirement`: `str | null`
- `industry_preference`: `list[str] | null`
- `keywords`: `list[str] | null`（当前 matching 主要用 skill/经验/教育/行业 + semantic；后续可用于 hr-agent）
- `job_summary`: `str | null`
- `job_title`: `str | null`（建议由前端或 LLM 抽取提供）

### 1.2 值域/规范化规则（建议前端与后端共同遵守）

为了让 matching 稳定（匹配服务会把 `required_skills`/`preferred_skills` 统一 lower-case），建议：
- 技能/行业列表输出为“语义上等价但不强制大小写一致”的文本，但后端入库前执行最少规范化：
  - 去首尾空格
  - 列表项去重（保留首次出现）
- `required_skills/preferred_skills/industry_preference` 中的字符串尽量不要包含过长短语，优先输出“可匹配的短 token”（比如 `FastAPI` 而不是一整句要求）
- `min_years`：只输出数字（例如 `3`），不要输出 `3年以上`
- `education_requirement`：输出“学历短语”（例如 `本科及以上` / `硕士及以上` / `学历不限`）

---

## 2. 前端填表默认路径（不走 JD parser）

### 2.1 前端提交给后端的字段映射

后端 Job 接口目前为：
- `POST /api/v1/jobs`：`JobCreate`
  - `title`
  - `raw_jd_text`（可选）
  - `structured`（可选）
  - `status`

建议前端默认路径：
- 直接填好表单并组装 `structured`（即 `JobStructured`）
- `title` 用 `job_name/job_title` 展示名
- `raw_jd_text` 默认不传（或传空/null）
- 将 `auto_analyze=false` 或者不依赖其行为（更稳妥）

这样后端就不会触发旧的规则解析路径（避免重复/不确定解析）。

### 2.2 前端表单元素如何设计（按 canonical json 来）

建议前端字段组：
- Job 基本信息：
  - `job_name`（-> `JobCreate.title`）
  - `job_title`（可选 -> `structured.job_title`）
  - `education_requirement`（-> `structured.education_requirement`）
  - `min_years`（-> `structured.min_years`）
  - `job_summary`（-> `structured.job_summary`，可选）
- 技能模块：
  - `required_skill[]`（-> `structured.required_skills`）
  - `preferred_skill[]`（-> `structured.preferred_skills`）
- 行业模块：
  - `preferred_industry[]` 或 `industry_preference[]`（-> `structured.industry_preference`）
- 关键词模块（可选）：
  - `keywords[]`（-> `structured.keywords`）

---

## 3. 上传 JD 备选路径（image/pdf/txt/word/doc -> 抽取标准 JSON）

当用户上传 JD 文件时，需要把文件内容抽取成：
1) `raw_jd_text`（全文/尽可能完整的文本）
2) `structured`（canonical json）

最终创建 Job 时：
- 优先传 `structured`（抽取结果）
- `title` 可以由：
  - 前端页面给默认值
  - 或抽取后从 `job_summary/job_title` 派生

### 3.1 抽取管道

建议为 Job 增加一个“文件上传入口”，类似简历：
- `POST /api/v1/jobs/upload`（multipart）
  - `file`
  - 可选：`title`/`status`/`auto_analyze`/`source` 等

处理步骤（实现为 service + parser pipeline）：
- 保存上传文件到临时目录
- 文件类型分流（固定）：
  - `pdf`、`image`：**PaddleOCR + LLM** 路径
    - PaddleOCR 负责文本块识别/版面文本还原
    - LLM 负责从 OCR 文本抽取 canonical `structured` JSON
  - `txt`、`word/doc/docx`：**Python 提取 + LLM** 路径
    - Python 工具先提取纯文本（如 txt 直接读、docx 用 python-docx、doc 用可用转换器）
    - LLM 再把文本结构化为 canonical `structured` JSON
- 标准化输出：
  - 输出 `raw_jd_text`（提取后的正文）
  - 输出 `structured`（canonical）

说明：
- 本计划不再使用规则 parser（`jd_analyzer`）路径，统一采用 LLM 结构化抽取。
- API 契约统一：上传与填表最终都产出同一份 `structured`。

---

## 4. 后端要落地的执行清单（按阶段）

### 阶段 1：固化 schema 与字段名（最小改动）
1. 确认前端 `structured` 字段严格使用 `snake_case`，并与 `JobStructured` 完全一致。
2. 更新/补齐文档：把 canonical json schema 写清楚（本文件就是）。
3. 在后端入参校验处：
   - 将 `JobCreate.structured` 明确当作 `JobStructured` 验证对象（可以先不改代码，只要求前端输出满足字段）。

### 阶段 2：前端默认路径接入（不走 parser）
1. 前端把表单值组装为 `structured`，并提交 `POST /api/v1/jobs`：
   - `title`
   - `structured`
   - `status`（建议直接设 `active` 或与状态机约定一致）
   - `raw_jd_text` 留空
2. 后端现有 `auto_analyze` 不触发规则解析（因为 `structured` 已提供）。

### 阶段 3：新增 Job 上传入口（固定两条抽取链）
1. 新增 API：`POST /api/v1/jobs/upload`（multipart）
2. 实现服务（按 MIME/后缀分流）：
   - `pdf/image` -> PaddleOCR -> OCR text -> LLM -> canonical `structured`
   - `txt/word/doc/docx` -> Python 提取 text -> LLM -> canonical `structured`
   - 两条路径都输出 `raw_jd_text`
3. 创建 Job：
   - `title` 取前端或抽取结果
   - `structured` 写入
   - `raw_jd_text` 可选写入（用于调试/可重试）
4. 入队 `job_profile_index_build`（你已有 job profile index build 任务链，直接复用即可）。

### 阶段 4：测试与验收
1. 单测：
   - 表单默认路径（structured 提供）-> 不触发解析链
   - 上传 PDF/图片 -> 走 PaddleOCR + LLM -> 输出 canonical json 并创建 job
   - 上传 txt/word/doc/docx -> 走 Python 提取 + LLM -> 输出 canonical json 并创建 job
2. 集成测试：
   - 上传 -> job_profile_index_build -> `/matching/run` 能成功拿到 semantic 分数（即 job 向量已可用）。

---

## 5. 验收标准（Acceptance Criteria）

1. 前端填表生成的 canonical `structured`：
   - 字段名完全匹配 `JobStructured`
   - 数据类型正确（list[int]/list[str]/int/str）
2. 任何上传文件（image/pdf/txt/word/doc/docx）最终都会得到：
   - `raw_jd_text`（非空，除非源文件本身不可读）
   - `structured` 满足 canonical json schema
3. backend 在默认填表路径下不会触发任何规则解析器链路（`jd_analyzer` 已下线）。
4. 上传路径创建的 Job 会进入 job profile index build 任务链，最终 matching 的 semantic_score 不再是 `not_indexed`。

---

## 6. 计划执行顺序（建议你按这个来改代码）

1. 固化并对齐 canonical schema（本文件先确认）
2. 前端先对接“默认填表 -> structured -> POST /jobs”
3. 后端补 “Job upload” 入口 + 文件->structured 抽取
4. 加 2-3 个代表性 JD 文件做 smoke test（不同格式）
5. 下线并删除 `jd_analyzer` 相关调用与文档，保持“前端 structured + 上传两条 LLM 抽取链”唯一口径

