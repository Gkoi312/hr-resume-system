"""
Seed demo data: 3 JDs + 10 candidates for the Demo UI.

Run:  python scripts/seed_demo_data.py
(requires DATABASE_URL pointing to a running PostgreSQL with pgvector)
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import init_db
from app.database.repository.job_repository import JobRepository
from app.database.repository.candidate_repository import CandidateRepository
from app import statuses


# ── Mock JDs ────────────────────────────────────────────────────────────

MOCK_JOBS = [
    dict(
        title="后端开发实习生",
        raw_jd_text=(
            "【岗位职责】\n"
            "1. 参与公司核心业务后端 API 的设计与开发\n"
            "2. 负责数据库设计与优化，编写高质量 SQL\n"
            "3. 参与系统架构讨论，输出技术方案文档\n"
            "4. 与前端、产品协作，确保需求高质量交付\n\n"
            "【任职要求】\n"
            "1. 本科及以上学历，计算机相关专业\n"
            "2. 熟悉 Python，了解 FastAPI/Django/Flask 等框架\n"
            "3. 熟悉 PostgreSQL/MySQL 等关系型数据库\n"
            "4. 了解 Docker、Git 等基本工具\n"
            "5. 有实际项目经验者优先"
        ),
        structured=dict(
            job_title="后端开发实习生",
            required_skills=["Python", "SQL", "Git"],
            preferred_skills=["FastAPI", "Docker", "PostgreSQL", "Redis"],
            responsibilities=[
                "参与后端 API 设计与开发",
                "数据库设计与优化",
                "参与系统架构讨论",
                "撰写技术方案文档",
            ],
            min_years=0,
            education_requirement="本科",
            industry_preference=["互联网"],
            keywords=["后端", "Python", "API", "数据库"],
            job_summary="参与核心业务后端系统开发，适合有 Python 基础的计算机专业学生。",
        ),
    ),
    dict(
        title="前端开发实习生",
        raw_jd_text=(
            "【岗位职责】\n"
            "1. 负责 Web 前端页面开发与维护\n"
            "2. 参与组件库建设与前端工程化优化\n"
            "3. 与后端协作完成接口联调\n"
            "4. 关注用户体验与页面性能\n\n"
            "【任职要求】\n"
            "1. 本科及以上学历\n"
            "2. 熟悉 HTML/CSS/JavaScript，了解 React 或 Vue\n"
            "3. 了解前端工程化工具（Webpack/Vite）\n"
            "4. 有良好的审美和用户体验意识"
        ),
        structured=dict(
            job_title="前端开发实习生",
            required_skills=["JavaScript", "HTML", "CSS", "React"],
            preferred_skills=["TypeScript", "Vue", "Vite", "Tailwind CSS"],
            responsibilities=[
                "Web 前端页面开发与维护",
                "组件库建设与前端工程化优化",
                "与后端协作完成接口联调",
                "关注用户体验与页面性能",
            ],
            min_years=0,
            education_requirement="本科",
            industry_preference=["互联网"],
            keywords=["前端", "React", "JavaScript", "Web"],
            job_summary="参与公司 Web 产品前端开发，适合有 React/Vue 经验的学生。",
        ),
    ),
    dict(
        title="数据分析实习生",
        raw_jd_text=(
            "【岗位职责】\n"
            "1. 负责业务数据的提取、清洗与分析\n"
            "2. 搭建数据看板与自动化报表\n"
            "3. 协助完成 A/B 实验设计与分析\n"
            "4. 为产品和运营提供数据驱动的决策建议\n\n"
            "【任职要求】\n"
            "1. 硕士及以上学历，统计学/数学/计算机相关专业\n"
            "2. 熟练使用 Python（pandas/numpy）和 SQL\n"
            "3. 了解基本的统计方法与机器学习算法\n"
            "4. 有数据可视化经验（Tableau/Echarts）优先"
        ),
        structured=dict(
            job_title="数据分析实习生",
            required_skills=["Python", "SQL", "数据分析"],
            preferred_skills=["机器学习", "Tableau", "A/B测试", "pandas"],
            responsibilities=[
                "业务数据提取、清洗与分析",
                "搭建数据看板与自动化报表",
                "A/B 实验设计与分析",
                "提供数据驱动的决策建议",
            ],
            min_years=0,
            education_requirement="硕士",
            industry_preference=["互联网", "金融"],
            keywords=["数据分析", "Python", "SQL", "机器学习"],
            job_summary="参与数据团队的分析工作，适合统计/计算机背景的研究生。",
        ),
    ),
]

# ── Mock Candidates ─────────────────────────────────────────────────────

MOCK_CANDIDATES = [
    dict(
        name="张三",
        email="zhangsan@example.com",
        phone="13800001001",
        education=[
            dict(school="清华大学", degree="硕士", major="计算机科学与技术"),
        ],
        work_experience=[
            dict(
                company="字节跳动",
                title="后端开发实习生",
                duration="2025.06 - 2025.09",
                description="参与广告投放系统后端开发，使用 Go 和 Python 完成 API 接口设计与数据库优化。",
            ),
        ],
        skills=["Python", "Go", "SQL", "Docker", "PostgreSQL", "Git"],
        projects=[
            dict(name="分布式缓存系统", description="基于 Redis Cluster 实现的分布式缓存，支持故障转移和数据分片"),
            dict(name="微服务 API 网关", description="Kong 插件开发，实现自定义鉴权与限流逻辑"),
        ],
        years_of_experience=0,
        summary="清华计算机硕士，有字节跳动后端实习经验，技术栈 Python/Go，具分布式系统项目经验。",
        direction_tags=["后端", "分布式", "云计算"],
    ),
    dict(
        name="李四",
        email="lisi@example.com",
        phone="13800001002",
        education=[
            dict(school="北京大学", degree="本科", major="软件工程"),
        ],
        work_experience=[
            dict(
                company="美团",
                title="前端开发实习生",
                duration="2025.03 - 2025.08",
                description="负责商家后台管理系统的前端开发，使用 React + TypeScript，优化页面加载性能 30%。",
            ),
        ],
        skills=["JavaScript", "TypeScript", "React", "Vue", "HTML", "CSS", "Git"],
        projects=[
            dict(name="在线协作白板", description="基于 Canvas API 和 WebSocket 的实时协作绘图工具"),
            dict(name="个人博客系统", description="使用 Next.js + MDX 构建的静态博客，支持暗色模式和全文搜索"),
        ],
        years_of_experience=0,
        summary="北大软工本科生，有美团前端实习经验，技术栈 React/TypeScript/Vue。",
        direction_tags=["前端", "全栈", "产品"],
    ),
    dict(
        name="王五",
        email="wangwu@example.com",
        phone="13800001003",
        education=[
            dict(school="浙江大学", degree="硕士", major="统计学"),
        ],
        work_experience=[
            dict(
                company="阿里巴巴",
                title="数据分析实习生",
                duration="2025.07 - 2025.12",
                description="负责淘宝推荐系统的数据分析与 A/B 实验评估，使用 Python + SQL 完成特征工程和效果评估。",
            ),
        ],
        skills=["Python", "SQL", "pandas", "机器学习", "数据分析", "Tableau"],
        projects=[
            dict(name="用户流失预测模型", description="使用 XGBoost 构建用户流失预测模型，AUC 达到 0.85"),
            dict(name="电商数据可视化看板", description="基于 Streamlit + Plotly 构建的实时销售数据监控看板"),
        ],
        years_of_experience=0,
        summary="浙大统计学硕士，有阿里数据分析实习经验，擅长 Python/SQL 和机器学习。",
        direction_tags=["数据分析", "机器学习", "数据科学"],
    ),
    dict(
        name="赵六",
        email="zhaoliu@example.com",
        phone="13800001004",
        education=[
            dict(school="上海交通大学", degree="本科", major="计算机科学"),
        ],
        work_experience=[
            dict(
                company="腾讯",
                title="后端开发实习生",
                duration="2025.04 - 2025.09",
                description="参与微信支付系统后端开发，使用 C++ 和 Python 实现交易对账和异常检测模块。",
            ),
        ],
        skills=["Python", "C++", "SQL", "Redis", "Git", "Linux"],
        projects=[
            dict(name="简易数据库引擎", description="从零实现 B+ 树索引和 SQL 解析器的轻量数据库"),
            dict(name="高并发秒杀系统", description="基于消息队列 + Redis 的秒杀系统，支持 10000 QPS"),
        ],
        years_of_experience=0,
        summary="上交计算机本科生，有腾讯后端实习经验，C++/Python 技术栈，对系统编程有浓厚兴趣。",
        direction_tags=["后端", "系统", "基础设施"],
    ),
    dict(
        name="孙七",
        email="sunqi@example.com",
        phone="13800001005",
        education=[
            dict(school="华中科技大学", degree="本科", major="计算机科学与技术"),
        ],
        work_experience=[],
        skills=["Python", "JavaScript", "Django", "React", "SQL", "Git"],
        projects=[
            dict(name="校园二手交易平台", description="Django + React 全栈项目，支持发布、搜索、聊天、评价功能"),
            dict(name="课程评价网站", description="爬取教务系统数据，提供课程评分与评价检索"),
        ],
        years_of_experience=0,
        summary="华科本科生，全栈方向，独立完成多个校园平台类项目。",
        direction_tags=["全栈", "后端", "前端"],
    ),
    dict(
        name="周八",
        email="zhouba@example.com",
        phone="13800001006",
        education=[
            dict(school="南京大学", degree="硕士", major="软件工程"),
        ],
        work_experience=[
            dict(
                company="华为",
                title="软件开发实习生",
                duration="2024.12 - 2025.06",
                description="参与鸿蒙系统应用框架开发，使用 C/C++ 和 Python 实现系统服务组件。",
            ),
        ],
        skills=["C++", "Python", "Java", "Linux", "Git", "Docker"],
        projects=[
            dict(name="操作系统内核实验", description="实现进程调度、内存管理、文件系统等核心模块"),
            dict(name="分布式 KV 存储", description="基于 Raft 共识算法的分布式键值存储系统"),
        ],
        years_of_experience=0,
        summary="南大软工硕士，有华为实习经验，系统编程功底扎实。",
        direction_tags=["系统", "后端", "基础设施"],
    ),
    dict(
        name="吴九",
        email="wujiu@example.com",
        phone="13800001007",
        education=[
            dict(school="普通学院", degree="大专", major="计算机应用"),
        ],
        work_experience=[
            dict(
                company="本地科技公司",
                title="Java 开发",
                duration="2025.03 - 2025.09",
                description="负责公司内部 OA 系统维护和功能开发。",
            ),
        ],
        skills=["Java", "Spring Boot", "MySQL", "HTML", "CSS"],
        projects=[
            dict(name="企业OA系统", description="Spring Boot + Thymeleaf 企业办公系统"),
        ],
        years_of_experience=0,
        summary="大专学历，有 Java 开发经验。",
        direction_tags=["Java", "企业级"],
    ),
    dict(
        name="郑十",
        email="zhengshi@example.com",
        phone="13800001008",
        education=[
            dict(school="复旦大学", degree="本科", major="数据科学"),
        ],
        work_experience=[
            dict(
                company="小红书",
                title="数据工程实习生",
                duration="2025.06 - 2025.09",
                description="负责内容推荐相关的数据 pipeline 开发与指标计算，使用 Python + Spark + Hive。",
            ),
        ],
        skills=["Python", "SQL", "Spark", "Hive", "数据分析", "pandas"],
        projects=[
            dict(name="实时日志分析平台", description="基于 Kafka + Spark Streaming 的实时日志处理系统"),
            dict(name="推荐算法评估框架", description="自动化评估推荐模型 offline/online 指标的框架"),
        ],
        years_of_experience=0,
        summary="复旦数据科学本科生，有小红书数据工程实习经验，大数据技术栈。",
        direction_tags=["数据分析", "数据工程", "推荐系统"],
    ),
    dict(
        name="陈十一",
        email="chenshiyi@example.com",
        phone="13800001009",
        education=[
            dict(school="北京航空航天大学", degree="硕士", major="计算机科学与技术"),
        ],
        work_experience=[
            dict(
                company="百度",
                title="前端开发实习生",
                duration="2025.05 - 2025.10",
                description="参与搜索产品的前端开发，使用 React + TypeScript，负责搜索结果页的交互优化。",
            ),
        ],
        skills=["TypeScript", "React", "JavaScript", "CSS", "Vite", "Git", "Node.js"],
        projects=[
            dict(name="低代码表单设计器", description="拖拽式表单设计器，支持自定义组件和 JSON Schema 导出"),
            dict(name="前端监控 SDK", description="实现页面性能、错误、用户行为的自动采集与上报"),
        ],
        years_of_experience=0,
        summary="北航计算机硕士，有百度前端实习经验，TypeScript/React 技术栈。",
        direction_tags=["前端", "全栈", "工具"],
    ),
    dict(
        name="刘十二",
        email="liushier@example.com",
        phone="13800001010",
        education=[
            dict(school="武汉大学", degree="本科", major="计算机科学与技术"),
        ],
        work_experience=[],
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Vue", "Git"],
        projects=[
            dict(name="HR 简历筛选系统", description="FastAPI + PostgreSQL 构建的 AI 简历筛选后端，支持多轴语义匹配与向量检索"),
            dict(name="开源贡献", description="为多个 Python 开源项目贡献代码，包括 FastAPI 插件和工具库"),
        ],
        years_of_experience=0,
        summary="武大本科生，全栈方向，独立开发过完整的后端系统项目，有开源贡献。",
        direction_tags=["后端", "全栈", "开源"],
    ),
]


# ── Main ─────────────────────────────────────────────────────────────────

async def seed():
    print("Initializing database tables...")
    await init_db()
    print("Done.\n")

    job_repo = JobRepository()
    cand_repo = CandidateRepository()

    # Check if already seeded
    existing = await job_repo.list(limit=1)
    if existing:
        print(f"Database already has {len(existing)}+ jobs. Skipping seed.")
        print("Run scripts/reset_database.py first if you need to re-seed.")
        return

    # Create jobs
    print("Creating demo JDs...")
    for jd in MOCK_JOBS:
        job = await job_repo.create(**jd, status=statuses.JOB_STATUS_ACTIVE)
        print(f"  ✅ {job.title}")

    # Create candidates
    print(f"\nCreating {len(MOCK_CANDIDATES)} demo candidates...")
    for cand in MOCK_CANDIDATES:
        c = await cand_repo.create(**cand)
        print(f"  ✅ {c.name}  ({c.email})")

    print(f"\n🎉 Done! Seeded {len(MOCK_JOBS)} jobs + {len(MOCK_CANDIDATES)} candidates.")
    print("   Open Demo UI → 选择岗位 → 运行匹配")


if __name__ == "__main__":
    asyncio.run(seed())
