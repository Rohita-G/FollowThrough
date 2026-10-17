"""FollowThrough Core: Evidence-backed conversational task extraction, classification, and ledger tracking."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from followthrough_core.task_extractor import extract_candidate_tasks
from followthrough_core.task_store import TaskStore
from followthrough_core.followup_linker import FollowupLinker
from followthrough_core.calendar_sync import generate_google_calendar_url, generate_ics_content
from followthrough_core.long_message_classifier import classify_long_message
from followthrough_core.classifier import DistilBERTClassifier, TFIDFBaselineClassifier

__all__ = [
    "extract_candidate_tasks",
    "TaskStore",
    "FollowupLinker",
    "classify_long_message",
    "DistilBERTClassifier",
    "TFIDFBaselineClassifier",
]
