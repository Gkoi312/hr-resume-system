# tests/test_resume_layer1.py
"""Layer 1 segmentation and parsing."""

from __future__ import annotations

from app.parsers.resume_parser.resume_rule_layer1.headers import SegmentKind
from app.parsers.resume_parser.resume_rule_layer1.segmenter import segment_resume
from app.parsers.resume_parser import parse_resume_text
from app.parsers.resume_parser.candidate_profile_builder import get_candidate_bind_for_resume
from app.schemas.resume_frame import ResumeLayer1Document


def test_segment_resume_standard_order():
    text = (
        "李四\n"
        "13800001111\n"
        "lisi@example.com\n"
        "\n"
        "教育背景\n"
        "北京大学 本科 计算机 2018-2022\n"
        "\n"
        "工作经历\n"
        "2022-2024 某公司 工程师\n"
        "• 负责后端\n"
        "\n"
        "项目经历\n"
        "2023 某某系统\n"
        "\n"
        "专业技能\n"
        "Python, Go\n"
    )
    r = segment_resume(text)
    assert "教育背景" in r.segments[SegmentKind.EDUCATION].raw_block
    assert "工作经历" in r.segments[SegmentKind.WORK].raw_block
    assert "项目经历" in r.segments[SegmentKind.PROJECTS].raw_block
    assert "专业技能" in r.segments[SegmentKind.ADDITIONAL].raw_block
    assert "自我评价" not in r.segments[SegmentKind.BASIC].raw_block
    kinds = {m["kind"] for m in r.subsection_markers}
    assert "skills" in kinds


def test_segment_self_eval_not_in_basic():
    text = (
        "王五\n"
        "13900002222\n"
        "自我评价\n"
        "乐观开朗，善于沟通。\n"
        "教育背景\n"
        "清华大学 硕士\n"
    )
    r = segment_resume(text)
    basic = r.segments[SegmentKind.BASIC].raw_block
    assert "自我评价" not in basic
    assert "乐观开朗" not in basic
    add = r.segments[SegmentKind.ADDITIONAL].raw_block
    assert "自我评价" in add
    kinds = [m["kind"] for m in r.subsection_markers]
    assert "self_introduction" in kinds


def test_parse_resume_layer1_validates():
    text = "赵六\nzhao@ex.com\n专业技能\nJava\n"
    out = parse_resume_text(text, document_id="d1", candidate_id="c1", file_name="r.txt")
    ResumeLayer1Document.model_validate(
        {
            "parser_version": out["parser_version"],
            "document_meta": out["document_meta"],
            "layer_1_extracted": out["layer_1_extracted"],
        }
    )
    assert out["parser_version"] == "resume_v1"
    assert out["document_meta"]["document_id"] == "d1"
    assert out["layer_1_extracted"]["basic"]["email"] == "zhao@ex.com"
    skills = out["layer_1_extracted"]["skills"]
    assert "java" in skills


def test_parse_resume_bind_via_layer2_on_demand():
    text = "钱七\nqian@ex.com\n3年工作经验\n"
    out = parse_resume_text(text)
    assert "layer_2_normalized" not in out
    bind = get_candidate_bind_for_resume(out)
    assert bind.get("email") == "qian@ex.com"
    assert bind.get("years_of_experience") is None
