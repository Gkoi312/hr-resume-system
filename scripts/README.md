# 脚本说明

## 数据库初始化（首次运行前）

1. 在项目根目录创建 `.env`，配置 PostgreSQL 连接（已提供 `.env.example` 示例）：
   ```
   DATABASE_URL=postgresql+asyncpg://postgres:你的密码@localhost:5432/hr_resume
   ```

2. 在 PostgreSQL 中创建数据库（Windows，在项目根目录执行）：
   ```powershell
   $env:PGPASSWORD="20020312gjx"; psql -U postgres -h localhost -c "CREATE DATABASE hr_resume;"
   ```
   若数据库已存在可忽略报错。

## verify_second_priority.py（第二优先级验证）

用于验证「Candidate 主档 + 匹配统一读 Candidate + 解析不覆盖人工编辑」的完整流程。

### 前置条件

1. **数据库**：若本机未安装/未启动 PostgreSQL，可改用 SQLite（无需安装）：
   ```powershell
   # PowerShell（项目根目录）
   $env:DATABASE_URL = "sqlite+aiosqlite:///./hr_resume.db"
   ```
   参见项目根目录 `.env.example`。使用 SQLite 时会在当前目录生成 `hr_resume.db` 文件。

2. **后端已启动**（默认 `http://127.0.0.1:8000`）：
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Worker 已启动**（负责执行简历解析任务）：
   ```bash
   python -m app.workers.task_worker
   ```

4. 项目根目录下存在：
   - `jobs/`：至少一个 `.md` 岗位描述（如 `jobs/AI_solve.md`）
   - `resumes/`：至少一个 `.pdf` 简历（如 `resumes/产品实习生_华东师范大学_高嘉欣.pdf`）

### 运行方式

在**项目根目录**执行：

```bash
python scripts/verify_second_priority.py
```

指定后端地址时：

```bash
set BASE_URL=http://localhost:8000
python scripts/verify_second_priority.py
```

### 验证内容

1. 用 `jobs/` 下第一个 `.md` 创建岗位并调用 `retry-analyze`。
2. 用 `resumes/` 下第一个 `.pdf` 调用 `POST /tasks/resume-upload`，轮询任务完成，得到 `candidate_id`。
3. 调用 `POST /matching/run` 得到**第一次匹配分数**。
4. `PATCH /candidates/{id}` 修改 `skills`、`years_of_experience`。
5. 再次 `POST /matching/run`，得到**第二次匹配分数**；若与第一次不同，说明匹配读取的是 Candidate 主档。
6. 再次上传同一简历并绑定同一 candidate，轮询完成后 `GET /candidates/{id}`，确认 skills/years 仍为 PATCH 后的值，说明解析未覆盖人工编辑。
