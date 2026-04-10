# app/parsers/resume_llm/extract.py
"""Call LLM for layer1_simple_v1 JSON.

配置：环境变量前缀 RESUME_LLM_*（见根目录 .env.example）。
推荐对接阿里百炼：PROVIDER=openai_compatible，BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1，MODEL=qwen-plus 等。
"""

from __future__ import annotations

import logging
import os
import unicodedata
from typing import Any, Dict

from app.llm.chat_client import ChatLLMClient, LLMClientError
from app.parsers.pdf_page_images import vision_png_base64_pages_from_file
from app.parsers.resume_parser.resume_llm_layer1.prompts import (
    RESUME_LAYER1_SIMPLE_SYSTEM,
    build_user_prompt,
    build_user_prompt_vision,
)
from app.parsers.resume_parser.resume_llm_layer1.simple_postprocess import (
    apply_garbled_input_heuristic,
    coerce_layer1_simple,
    compact_ungrounded_projects,
    compact_ungrounded_work,
    filter_list_fields_against_source,
    validate_layer1_simple_shape,
)

logger = logging.getLogger(__name__)

resume_llm_client = ChatLLMClient(env_prefix="RESUME_LLM_")
resume_vllm_client = ChatLLMClient(env_prefix="RESUME_VLLM_")


def resume_llm_enabled() -> bool:
    """True when RESUME_LLM_ENABLED is 1/true/yes (case-insensitive)."""
    v = (os.getenv("RESUME_LLM_ENABLED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def resume_llm_vision_enabled() -> bool:
    """True when visual extraction is enabled (RESUME_LLM_VISION or RESUME_VLLM_ENABLED)."""
    v1 = (os.getenv("RESUME_LLM_VISION") or "").strip().lower()
    v2 = (os.getenv("RESUME_VLLM_ENABLED") or "").strip().lower()
    return (v1 in ("1", "true", "yes", "on")) or (v2 in ("1", "true", "yes", "on"))


def _has_vllm_override() -> bool:
    return any(
        (os.getenv(k) or "").strip()
        for k in (
            "RESUME_VLLM_PROVIDER",
            "RESUME_VLLM_API_KEY",
            "RESUME_VLLM_BASE_URL",
            "RESUME_VLLM_MODEL",
        )
    )


def _validate_simple_top(obj: Dict[str, Any]) -> None:
    if obj.get("schema_version") != "layer1_simple_v1":
        raise LLMClientError(
            f"Unexpected schema_version: {obj.get('schema_version')!r}, expected layer1_simple_v1"
        )
    for key in (
        "basic",
        "education",
        "work_experience",
        "projects",
        "additional",
        "warnings",
    ):
        if key not in obj:
            raise LLMClientError(f"Missing key in LLM output: {key}")


async def extract_resume_simple_json(resume_text: str) -> Dict[str, Any]:
    """
    Returns full simple JSON (schema_version + basic + ... + warnings).
    Normalizes text with NFKC before sending.
    """
    normalized = unicodedata.normalize("NFKC", resume_text or "")
    user = build_user_prompt(normalized)
    obj = await resume_llm_client.generate_json(
        system_prompt=RESUME_LAYER1_SIMPLE_SYSTEM,
        user_prompt=user,
    )
    _validate_simple_top(obj)
    obj = coerce_layer1_simple(obj)
    apply_garbled_input_heuristic(obj, normalized)
    validate_layer1_simple_shape(obj)
    filter_list_fields_against_source(obj, normalized)
    compact_ungrounded_work(obj, normalized)
    compact_ungrounded_projects(obj, normalized)
    return obj


async def extract_resume_simple_json_vision(
    resume_text: str,
    *,
    file_bytes: bytes,
    file_name: str = "resume.pdf",
) -> Dict[str, Any]:
    """
    Vision path: PDF or image bytes -> PNG base64 pages -> vision LLM -> layer1_simple_v1 JSON.

    Important: In vision mode we *may* get empty/unreliable text layer, so grounding-based
    filtering (substrings) is only applied when resume_text has content.
    """
    normalized = unicodedata.normalize("NFKC", resume_text or "")

    max_pages = int(os.getenv("RESUME_LLM_VISION_MAX_PAGES") or "2")
    dpi = float(os.getenv("RESUME_LLM_VISION_DPI") or "150")

    images = vision_png_base64_pages_from_file(
        file_bytes,
        file_name,
        max_pages=max_pages,
        dpi=dpi,
    )
    if not images:
        raise LLMClientError("Vision rasterize produced no images")

    user = build_user_prompt_vision(normalized, page_count=len(images))
    client = resume_vllm_client if _has_vllm_override() else resume_llm_client
    obj = await client.generate_json_with_images(
        system_prompt=RESUME_LAYER1_SIMPLE_SYSTEM,
        user_text=user,
        images_png_base64=images,
    )
    _validate_simple_top(obj)
    obj = coerce_layer1_simple(obj)
    validate_layer1_simple_shape(obj)

    # Only apply substring grounding when we actually have a usable text layer.
    # Otherwise we'd wipe VLM descriptions/skills because source_text is empty.
    if normalized.strip():
        filter_list_fields_against_source(obj, normalized)
        compact_ungrounded_work(obj, normalized)
        compact_ungrounded_projects(obj, normalized)

    return obj
