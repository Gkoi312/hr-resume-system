# app/parsers/resume_paddle/ppstructure_client.py
"""Lazy singleton PPStructureV3 + predict -> list of page JSON dicts."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Before any transitive ``import requests`` (Paddle / httpx chains), silence the
# noisy urllib3/chardet version mismatch warning seen on many Windows conda envs.
warnings.filterwarnings("ignore", message="urllib3", category=Warning)

logger = logging.getLogger(__name__)

_pipeline = None
_paddle_env_done = False


def _paddle_quiet_enabled() -> bool:
    """Default on: fewer Paddle/PaddleX checks and console warnings during OCR."""
    v = (os.getenv("RESUME_PADDLE_QUIET") or "1").strip().lower()
    return v not in ("0", "false", "no", "off", "n")


def ensure_paddle_runtime_env() -> None:
    """
    Apply env + warning filters before any ``paddleocr`` / PaddleX import.

    - Skips model-host connectivity checks when models are already under ``~/.paddlex`` (offline-friendly).
    - With ``RESUME_PADDLE_QUIET`` default on: tones down ccache / requests version UserWarnings and
      Paddle-related log spam. Set ``RESUME_PADDLE_QUIET=0`` to restore verbose diagnostics.
    """
    global _paddle_env_done
    if _paddle_env_done:
        return
    _paddle_env_done = True

    # PaddleX: skip hub/source reachability check (use local cache only).
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Legacy / alternate name seen in some PaddleOCR + PaddleX stacks.
    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

    if not _paddle_quiet_enabled():
        return

    warnings.filterwarnings(
        "ignore",
        message=".*[Cc]cache.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*ccache.*",
        category=UserWarning,
    )


def _silence_paddle_loggers() -> None:
    if not _paddle_quiet_enabled():
        return
    for name in (
        "paddlex",
        "paddle",
        "ppocr",
        "paddleocr",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


ensure_paddle_runtime_env()


def paddle_ppstructure_enabled() -> bool:
    v = (os.getenv("RESUME_PADDLE_ENABLED") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _get_device() -> str:
    return (os.getenv("RESUME_PADDLE_DEVICE") or "gpu").strip().lower()


def get_ppstructure_pipeline():
    """Return shared PPStructureV3 instance (import on first use)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if not paddle_ppstructure_enabled():
        raise RuntimeError("Paddle PP-Structure disabled (RESUME_PADDLE_ENABLED=0)")
    ensure_paddle_runtime_env()
    from paddleocr import PPStructureV3

    _pipeline = PPStructureV3(device=_get_device())
    _silence_paddle_loggers()
    return _pipeline


def _result_to_dicts(results) -> List[Dict[str, Any]]:
    """Materialize paddle results to JSON-serializable dicts via temp save."""
    out: List[Dict[str, Any]] = []
    if not results:
        return out
    with tempfile.TemporaryDirectory(prefix="ppstruct_") as td:
        root = Path(td)
        for i, res in enumerate(results):
            sub = root / str(i)
            sub.mkdir(parents=True, exist_ok=True)
            res.save_to_json(str(sub))
            json_files = sorted(sub.glob("*.json"))
            if not json_files:
                json_files = sorted(sub.rglob("*.json"))
            for jf in json_files:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "parsing_res_list" in data:
                        out.append(data)
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning("Paddle JSON read failed %s: %s", jf, e)
    return out


def predict_file_bytes_to_page_dicts(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Run PPStructureV3 on a file (PDF or image). Returns one dict per page (or single page).
    """
    if not file_bytes:
        return []
    suf = (Path(filename).suffix or "").lower() or ".pdf"
    pipeline = get_ppstructure_pipeline()
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        results = pipeline.predict(tmp_path)
        return _result_to_dicts(results)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def try_predict_file_bytes(
    file_bytes: bytes,
    filename: str,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Run Paddle predict; returns (page_dicts, error_message).

    On import/runtime failure, returns (None, message) so callers can fallback.
    """
    try:
        pages = predict_file_bytes_to_page_dicts(file_bytes, filename)
        return pages, None
    except Exception as e:  # noqa: BLE001
        logger.exception("PPStructureV3 predict failed: %s", e)
        return None, str(e)
