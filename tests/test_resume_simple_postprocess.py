# tests/test_resume_simple_postprocess.py

from app.parsers.resume_parser.resume_llm_layer1.simple_postprocess import (
    coerce_layer1_simple,
    compact_ungrounded_projects,
    compact_ungrounded_work,
    filter_list_fields_against_source,
)


def test_coerce_aliases_education_and_project_name():
    raw = {
        "schema_version": "layer1_simple_v1",
        "basic": {"summary": "PM", "links": []},
        "education": [
            {
                "school": "华东师大",
                "degree": "本科",
                "field_of_study": "软件工程",
                "start_date": "2020",
                "end_date": "2024",
                "bullets": ["a"],
                "raw_block": "",
            }
        ],
        "work_experience": [
            {
                "company": "A公司",
                "title": "产品实习生",
                "start": "2024-01",
                "end": "2024-06",
                "bullets": [],
                "raw_block": "",
            }
        ],
        "projects": [{"title": "项目甲", "bullets": [], "raw_block": ""}],
        "additional": {"certifications": ["CET-6"]},
        "warnings": [],
    }
    obj = coerce_layer1_simple(raw)
    assert obj["education"][0]["major"] == "软件工程"
    assert obj["education"][0]["start"] == "2020"
    assert obj["education"][0]["raw_block"] == "a"
    assert obj["work_experience"][0]["job_type"] == "internship"
    assert obj["projects"][0]["project_name"] == "项目甲"
    assert obj["additional"]["certificates"] == ["CET-6"]


def test_filter_and_compact_drop_hallucinations():
    src = "Waybox Technology Limited 实习\n- 只做文档整理。\n"
    basic_empty = (
        {k: "" for k in ["name", "phone", "email", "location", "birth_text", "target_role"]}
        | {"links": [], "raw_block": ""}
    )
    obj = coerce_layer1_simple(
        {
            "schema_version": "layer1_simple_v1",
            "basic": basic_empty,
            "education": [],
            "work_experience": [
                {
                    "company": "Waybox Technology Limited",
                    "job_role": "实习生",
                    "job_type": "unknown",
                    "start": "",
                    "end": "",
                    "descriptions": [
                        "只做文档整理。",
                        "负责LLM微调与RAG架构设计并部署生产环境。",
                    ],
                    "raw_block": "",
                }
            ],
            "projects": [
                {
                    "project_name": "基于RAG的智能问答系统",
                    "role": "",
                    "start": "",
                    "end": "",
                    "descriptions": ["使用LangChain构建向量检索。"],
                    "raw_block": "",
                }
            ],
            "additional": {
                "skills": [],
                "languages": [],
                "certificates": [],
                "awards": [],
                "self_evaluation": "",
                "raw_block": "",
            },
            "warnings": [],
        }
    )
    filter_list_fields_against_source(obj, src)
    assert len(obj["work_experience"][0]["descriptions"]) == 1
    compact_ungrounded_work(obj, src)
    compact_ungrounded_projects(obj, src)
    assert len(obj["work_experience"]) == 1
    assert len(obj["projects"]) == 0
