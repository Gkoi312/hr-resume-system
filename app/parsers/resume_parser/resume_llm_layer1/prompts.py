# app/parsers/resume_llm/prompts.py
"""System prompt for layer1_simple_v1 (see json_design/layer1_llm_prompt.md)."""

RESUME_LAYER1_SIMPLE_SYSTEM = """你是简历解析器。仅依据输入文本/图片中的可见内容，输出唯一一个 JSON 对象。

必须输出 schema_version="layer1_simple_v1"，并严格包含这些顶层键：
- basic(name, phone, email, location, birth_text, target_role, links[], raw_block)
- education[](school, degree, major, start, end, raw_block)
- work_experience[](company, job_role, job_type, start, end, descriptions[], raw_block)
- projects[](project_name, role, start, end, descriptions[], raw_block)
- skills[]
- additional(languages[], certificates[], awards[], self_evaluation, raw_block)
- warnings[]

规则：
1) 禁止编造；无法确认就留空（标量 ""，列表 []）。
2) skills[] 每个条目必须是简短的“技术名/工具名/语言/框架/库”而不是能力描述句子；优先把“技术/技术栈/技能”这些词作为条目。每条不超过 10 个中/英文字符。
3) descriptions / raw_block 必须来自输入中的原文片段，逐字复制，不改写。descriptions是对实习|工作|项目的描述。
4) job_type 只能是 internship|fulltime|parttime|unknown（含“实习/实习生/intern”优先 internship）。
5) warnings 仅写技术/版式问题（如乱码、顺序混乱、多栏错位），不要写评价建议。
6) education.degree 推断规则：
   - 若原文明确写了“本科/大专/硕士/博士”，按原文填写。
   - 若原文没写 degree：仅一段 education 时优先填“本科”或“大专”；两段及以上时按时间线从早到晚优先填“本科 + 硕士/博士”。
   - 若学校明显是虚假杜撰的，则填“无法判断”。
   - 禁止留空 degree；在“大专/本科/硕士/博士/无法判断”里给最合理判断。
"""


def build_user_prompt(resume_text: str) -> str:
    body = (resume_text or "").strip()
    return (
        "请解析以下简历全文，输出 layer1_simple_v1 JSON（仅 JSON）。\n"
        "若字段缺失请留空，不要编造；descriptions/raw_block 请尽量逐字引用原文。\n\n"
        "若 education 的 degree 未显式给出，请按教育条目数量与时间线推断（至少填 本科/大专/硕士/博士 之一）。\n\n"
        f"【简历全文】\n{body}"
    )


def build_user_prompt_vision(
    resume_text: str,
    *,
    page_count: int,
) -> str:
    """User text for multimodal call: instructions + optional PDF text layer (may be empty)."""
    body = (resume_text or "").strip()
    head = (
        "请解析简历并输出 layer1_simple_v1 JSON（仅 JSON）。\n"
        f"已附上该简历前 {page_count} 页截图（按顺序）。请优先基于截图文字；"
        "descriptions/raw_block 请逐字引用可见文本，不要编造。\n"
        "同一经历在 work_experience 与 projects 中只能出现一次；无公司名或校园经历优先归入 projects。严格按照我的要求执行，不要遗漏任何要求，也不要违反任何要求。\n\n"
    )
    if body:
        return (
            f"{head}\n"
            "下列【简历全文】来自 PDF/DOC 文本层，可能有遗漏、乱序或与版式不一致，请与截图交叉核对后再输出。\n\n"
            f"【简历全文】\n{body}"
        )
    return (
        f"{head}\n"
        "未提供可用的机读文本层；请完全依据截图填写。若某页几乎无有效文字可忽略该页。"
    )
