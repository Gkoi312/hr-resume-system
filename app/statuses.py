"""Centralized status constants for core domain objects.

This keeps allowed status values in one place so that APIs, services,
schemas, and ORM models stay consistent.
"""

# Job statuses
JOB_STATUS_DRAFT = "draft"
JOB_STATUS_ANALYZING = "analyzing"
JOB_STATUS_ACTIVE = "active"
JOB_STATUS_FAILED = "failed"

# Resume statuses
RESUME_STATUS_UPLOADED = "uploaded"
RESUME_STATUS_EXTRACTING = "extracting"
RESUME_STATUS_PARSED = "parsed"
RESUME_STATUS_CANDIDATE_BOUND = "candidate_bound"
RESUME_STATUS_FAILED = "failed"

# Match statuses
MATCH_STATUS_PENDING = "pending"
MATCH_STATUS_RUNNING = "running"
MATCH_STATUS_COMPLETED = "completed"
MATCH_STATUS_FAILED = "failed"

# Task statuses
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

__all__ = [
    # Job
    "JOB_STATUS_DRAFT",
    "JOB_STATUS_ANALYZING",
    "JOB_STATUS_ACTIVE",
    "JOB_STATUS_FAILED",
    # Resume
    "RESUME_STATUS_UPLOADED",
    "RESUME_STATUS_EXTRACTING",
    "RESUME_STATUS_PARSED",
    "RESUME_STATUS_CANDIDATE_BOUND",
    "RESUME_STATUS_FAILED",
    # Match
    "MATCH_STATUS_PENDING",
    "MATCH_STATUS_RUNNING",
    "MATCH_STATUS_COMPLETED",
    "MATCH_STATUS_FAILED",
    # Task
    "TASK_STATUS_PENDING",
    "TASK_STATUS_RUNNING",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
]

