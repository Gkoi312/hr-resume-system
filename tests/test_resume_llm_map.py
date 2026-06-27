# tests/test_resume_llm_map.py
"""Map simple LLM JSON -> layer_1_extracted (no network)."""

from app.parsers.resume_parser.resume_llm_layer1.map_simple_to_layer1 import map_simple_to_layer1
from app.schemas.resume_frame import ResumeLayer1Document


def _minimal_simple() -> dict:
    return {
        "schema_version": "layer1_simple_v1",
        "basic": {
            "name": "张三",
            "phone": "13800000000",
            "email": "a@b.com",
            "location": "上海",
            "birth_text": "1990年1月",
            "target_role": "工程师",
            "links": ["https://github.com/z"],
            "raw_block": "张三 13800000000",
        },
        "education": [
            {
                "school": "某某大学",
                "degree": "本科",
                "major": "软件工程",
                "start": "2016",
                "end": "2020",
                "raw_block": "",
            }
        ],
        "work_experience": [
            {
                "company": "A公司",
                "job_role": "开发",
                "job_type": "internship",
                "start": "2020-01",
                "end": "2020-06",
                "descriptions": ["负责接口"],
                "raw_block": "",
            }
        ],
        "projects": [
            {
                "project_name": "项目X",
                "role": "",
                "start": "2019",
                "end": "2019",
                "descriptions": ["实现原型"],
                "raw_block": "",
            }
        ],
        "skills": ["Python"],
        "additional": {
            "languages": ["英语 CET-6"],
            "certificates": [],
            "awards": [],
            "self_evaluation": "积极主动。",
            "raw_block": "工具：Git",
        },
        "warnings": [],
    }


def test_map_simple_to_layer1_validates():
    layer1 = map_simple_to_layer1(_minimal_simple())
    doc = {
        "parser_version": "resume_v1",
        "document_meta": {
            "document_id": "d",
            "candidate_id": "c",
            "file_name": "r.txt",
            "file_type": "txt",
            "language": "zh",
            "parse_time": "2026-01-01T00:00:00Z",
            "text_extraction_method": "text",
            "ocr_used": False,
            "total_pages": None,
            "parse_status": "success",
            "warnings": [],
            "raw_text_preview": "",
        },
        "layer_1_extracted": layer1,
    }
    m = ResumeLayer1Document.model_validate(doc)
    assert m.layer_1_extracted.basic.name == "张三"
    assert m.layer_1_extracted.basic.phone == "13800000000"
    assert len(m.layer_1_extracted.education) == 1
    assert m.layer_1_extracted.education[0].school == "某某大学"
    assert len(m.layer_1_extracted.work_experience) == 1
    assert m.layer_1_extracted.work_experience[0].job_type == "internship"
    assert m.layer_1_extracted.work_experience[0].descriptions == ["负责接口"]
    assert len(m.layer_1_extracted.projects) == 1
    assert m.layer_1_extracted.skills == ["python"]
    assert m.layer_1_extracted.additional.self_evaluation == "积极主动。"
