# app/parsers/resume_parser/resume_rule_layer1
"""Rule-based segmentation + extraction -> layer_1_extracted (aligned with simple)."""

from app.parsers.resume_parser.resume_rule_layer1.assemble import build_layer1_document
from app.parsers.resume_parser.resume_rule_layer1.segmenter import segment_resume

__all__ = [
    "build_layer1_document",
    "segment_resume",
]
