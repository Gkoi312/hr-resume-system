# app/parsers/resume_paddle

from app.parsers.resume_paddle.flatten import flatten_ppstructure_pages
from app.parsers.resume_paddle.ppstructure_client import (
    get_ppstructure_pipeline,
    paddle_ppstructure_enabled,
    predict_file_bytes_to_page_dicts,
    try_predict_file_bytes,
)

__all__ = [
    "flatten_ppstructure_pages",
    "get_ppstructure_pipeline",
    "paddle_ppstructure_enabled",
    "predict_file_bytes_to_page_dicts",
    "try_predict_file_bytes",
]
