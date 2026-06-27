# Match 链重构设计（全新方案）

## 1. 设计目标

- 完全重做匹配链，不继承旧匹配逻辑。
- 实现“先筛选再排序再解释”的闭环。
- 支持后续快速调参：阈值、权重、规则均可配置。
- 最终输出要能被 HR 理解（有证据、有理由）。


## 2. 总体架构（Pipeline）

### Step 1：字段对齐（Schema Alignment）
- 对 job 和 candidate 建立统一 Canonical Schema。
- 所有来源（前端填表、解析器抽取、历史数据）都先映射到 Canonical，再进入匹配。

### Step 2：标准化（Normalization）
- 技能标准化：大小写、别名、同义词统一。
- 学历标准化：映射到有序等级。
- 年限标准化：统一数值口径（总年限、相关年限可分开）。
- 方向标准化：前端/后端/数据/算法/测开等统一标签。

### Step 3：硬性过滤（Gate）
- 学历、必备技能、最低年限先判断，不达标直接淘汰或进入灰名单。

### Step 4：多路语义匹配（Semantic Multi-Route）
- 技能语义匹配
- 项目/经历语义匹配
- 岗位方向语义匹配

### Step 5：融合打分 + 解释输出
- 融合结构化分和语义分得出总分。
- 输出证据、风险点、面试追问建议。


## 3. Canonical Schema（对齐目标）

### 3.1 Job Canonical
- `job_title`
- `required_skills`
- `preferred_skills`
- `responsibilities`
- `education_min`
- `min_years`
- `industry_tags`
- `job_summary`

### 3.2 Candidate Canonical
- `skills_explicit`
- `skills_inferred`
- `projects[]`
- `work/internship_experience[]`
- `highest_education`
- `years_of_experience`
- `industry_exposure`
- `role_summary`

### 3.3 对齐策略
- 缺失字段允许为空，但必须存在键位。
- 解析器未给出的字段，通过规则抽取或 LLM 补齐。
- 对齐过程落库，保留 `raw -> canonical` 映射日志，便于追查。


## 4. 硬性过滤策略（Gate）

### 4.1 学历 Gate
- JD 要求“本科及以上”时，本科以下直接过滤。
- 通过后在打分中做教育加分：博士 / 硕士 > 本科。

### 4.2 必备技能 Gate
- `required_skills` 采用“命中一项即可通过 Gate”的规则（当前业务决策）。
- 后续如需收紧到“全部命中”或“命中 K 项”，通过配置切换，不改主流程。

### 4.3 年限 Gate
- 校招生/实习生岗位：`min_years` 默认留空，不作为 Gate 条件。
- 社招岗位：若 JD 明确最低年限，低于阈值时可配置为过滤或灰名单。

### 4.4 Gate 输出
- `gate_pass`
- `gate_failed_reasons[]`
- `gate_soft_warnings[]`


## 5. 证据型语义匹配

## 5.1 技能语义匹配
用于解决关键词不重合但能力等价的问题，例如：
- JD：Python、机器学习、SQL
- Candidate：pandas/sklearn/xgboost、mysql、pytorch 项目

覆盖场景：
- 同义表达
- 上下位概念
- 工具到能力的反推
- 项目文本中的隐式技能识别

比较对象：
- `jd.required_skills + jd.preferred_skills`
vs
- `candidate.skills_explicit + candidate.skills_inferred + skill_evidence_chunks`

## 5.2 项目语义匹配
核心判断：“候选人是否做过和岗位职责相似的事”。

比较对象：
- `jd.responsibilities`
vs
- `candidate.projects[] + work_experience[] + internship_experience[]`

## 5.3 方向语义匹配
核心判断：“候选人的职能方向是否贴近岗位方向”。

比较对象：
- `jd.job_title + jd.job_summary`
vs
- `candidate.role_summa ry + direction_tags_from_experience`


## 6. 分块 Embedding 与检索

## 6.1 问题定义
单一向量会冲平字段权重，噪声会稀释强信号，不能承载多种匹配意图。

## 6.2 Job 侧分块
- `job_title_block`
- `required_skills_block`
- `preferred_skills_block`
- `responsibilities_block`
- `requirements_block`
- `job_summary_block`

## 6.3 Candidate 侧分块
- `skill_block_explicit`
- `skill_block_inferred`
- `project_block_i`（项目逐条）
- `work_block_i`（工作/实习逐条）
- `role_summary_block`

## 6.4 检索流程
- 每条语义路线独立召回 Top-K 块。
- 对 Top-K 做 rerank。
- 取最终证据块写入解释结果（带来源和相似度）。


## 7. 评分融合（新版本）

### 7.1 语义子分
- `skill_similarity`
- `project_similarity`
- `role_similarity`

`semantic_score = 0.35 * skill_similarity + 0.45 * project_similarity + 0.20 * role_similarity`

### 7.2 非语义结构化分
- `hard_requirement_score`
- `experience_score`
- `education_bonus_score`
- `industry_fit_score`

### 7.3 总分公式（建议初版）
`overall_score = 0.55 * semantic_score + 0.20 * hard_requirement_score + 0.10 * experience_score + 0.10 * education_bonus_score + 0.05 * industry_fit_score`

说明：
- 语义分主导，适配真实简历表达差异。
- 硬条件仍保留显式影响，避免“语义高但硬条件不满足”的误排前。
- 权重后续按标注集和线上反馈迭代。


## 8. 输出结果设计

### 8.1 机器可读字段
- `gate_pass`
- `overall_score`
- `semantic_score`
- `sub_scores`（skill/project/role/experience/education/industry）
- `decision`（recommend_interview / further_screening / not_recommended）

### 8.2 人可读解释
- `hard_requirements_met`
- `missing_requirements`
- `strong_signals`
- `risk_signals`
- `semantic_evidence[]`（文本、来源、相似度、理由）
- `interview_focus_points`


## 9. 实施阶段

### Phase 1：对齐与 Gate
- 完成 Canonical Schema。
- 完成技能/学历/方向词表与标准化。
- 上线学历、技能、年限 Gate。

### Phase 2：分块语义
- 上线分块 embedding。
- 上线三路语义打分和证据回填。

### Phase 3：融合与调优
- 建立标注样本与评估集。
- 做权重调优与阈值优化。
