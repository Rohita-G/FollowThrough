"""FollowThrough Data Pipeline: Ingestion, parsing, cleaning, threading, and manifest generation."""

from data_pipeline.download_enron import download_enron
from data_pipeline.select_emails import select_emails
from data_pipeline.parse_emails import parse_email
from data_pipeline.clean_emails import clean_body
from data_pipeline.build_threads import build_threads
from data_pipeline.build_corpus_manifest import build_manifest
from data_pipeline.generate_weak_labels import compute_weak_labels

__all__ = [
    "download_enron",
    "select_emails",
    "parse_email",
    "clean_body",
    "build_threads",
    "build_manifest",
    "compute_weak_labels",
]
