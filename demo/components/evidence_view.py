"""
Evidence view helpers — thin re-exports from match_card for semantic clarity.
"""
from components.match_card import (
    render_delivery_alignments,
    render_llm_quality,
    render_score_breakdown,
    render_semantic_evidence,
)

__all__ = [
    "render_score_breakdown",
    "render_semantic_evidence",
    "render_delivery_alignments",
    "render_llm_quality",
]
