import copy
import hashlib
import json
import unittest
from pathlib import Path

from scripts.validate_annotations import load_reference_index, validate_examples, validate_file

FIXTURES = Path(__file__).parent / "fixtures"
CONTEXT = FIXTURES / "synthetic_annotation_context"
EXAMPLES = FIXTURES / "synthetic_annotation_examples.v1.jsonl"
REPO_ROOT = Path(__file__).parents[1]
SMOKE_NAMES = ("enron_subset_smoke_threads", "enron_subset_interrupt_smoke_threads")
SMOKE_OUTPUTS_PRESENT = all(
    (REPO_ROOT / "data/processed" / smoke_name / filename).is_file()
    for smoke_name in SMOKE_NAMES
    for filename in ("messages.jsonl", "threads.jsonl")
) and all(
    (REPO_ROOT / "data/annotations" / candidate_name).is_file()
    for candidate_name in ("candidates.smoke.csv", "candidates.interrupt_smoke.csv")
)


def _fixture_data():
    messages, threads, reference_issues = load_reference_index(CONTEXT / "messages.jsonl", CONTEXT / "threads.jsonl")
    examples = [json.loads(line) for line in EXAMPLES.read_text(encoding="utf-8").splitlines() if line]
    return messages, threads, reference_issues, examples


def test_synthetic_worked_examples_validate_and_are_isolated_by_origin():
    messages, threads, reference_issues, examples = _fixture_data()
    report = validate_examples(examples, messages, threads, reference_issues)
    assert report["error_count"] == 0, report["issues"]
    assert report["example_count"] == 11
    assert report["origin_counts"] == {"real": 0, "synthetic": 11}
    assert report["complete_count"] == 11
    assert report["draft_count"] == 0


def test_duplicate_example_id_and_invalid_schema_types_are_rejected():
    messages, threads, reference_issues, examples = _fixture_data()
    invalid = copy.deepcopy(examples[:2])
    invalid[1]["example_id"] = invalid[0]["example_id"]
    invalid[1]["ambiguous"] = "false"
    invalid[1]["relation_labels"] = ["request"]
    report = validate_examples(invalid, messages, threads, reference_issues)
    codes = {issue["code"] for issue in report["issues"]}
    assert {"duplicate_example_id", "field_type", "relation_label"} <= codes


def test_evidence_offsets_and_changed_field_evidence_are_checked():
    messages, threads, reference_issues, examples = _fixture_data()
    invalid = copy.deepcopy(examples[2])
    invalid["evidence_spans"][0]["end_char"] = len(invalid["update_text"]) + 1
    invalid["updated_deadline_text"] = "May 9"
    report = validate_examples([invalid], messages, threads, reference_issues)
    codes = {issue["code"] for issue in report["issues"]}
    assert "evidence_bounds" in codes
    assert "changed_deadline_not_literal" in codes
    assert "changed_deadline_without_evidence" in codes


def test_ambiguity_requires_flag_reason_and_label_consistency():
    messages, threads, reference_issues, examples = _fixture_data()
    invalid = copy.deepcopy(examples[8])
    invalid["ambiguous"] = False
    invalid["ambiguity_reason"] = ""
    report = validate_examples([invalid], messages, threads, reference_issues)
    codes = {issue["code"] for issue in report["issues"]}
    assert "ambiguity_mismatch" in codes

    invalid["ambiguous"] = True
    report = validate_examples([invalid], messages, threads, reference_issues)
    codes = {issue["code"] for issue in report["issues"]}
    assert "ambiguity_reason" in codes


def test_task_provenance_must_be_earlier_and_in_same_thread():
    messages, threads, reference_issues, examples = _fixture_data()
    later_source = copy.deepcopy(examples[1])
    later_source["task_source_message_ids"] = ["SYN-COMMIT-02"]
    report = validate_examples([later_source], messages, threads, reference_issues)
    assert "task_source_not_earlier" in {issue["code"] for issue in report["issues"]}

    cross_thread = copy.deepcopy(examples[1])
    cross_thread["thread_id"] = "SYN-DATE"
    report = validate_examples([cross_thread], messages, threads, reference_issues)
    codes = {issue["code"] for issue in report["issues"]}
    assert {"update_thread_mismatch", "task_source_thread_mismatch"} <= codes


def test_blank_draft_is_distinct_from_a_completed_annotation():
    messages, threads, reference_issues, examples = _fixture_data()
    draft = copy.deepcopy(examples[1])
    draft.update({
        "annotation_status": "draft",
        "message_act_labels": [],
        "relation_labels": [],
        "evidence_spans": [],
        "ambiguous": False,
        "ambiguity_reason": "",
        "updated_owner_text": "",
        "updated_deadline_text": "",
        "updated_deadline_normalized": None,
        "annotator_id": "",
        "annotation_notes": "",
    })
    report = validate_examples([draft], messages, threads, reference_issues)
    assert report["error_count"] == 0, report["issues"]
    draft["message_act_labels"] = ["commitment"]
    report = validate_examples([draft], messages, threads, reference_issues)
    assert "draft_not_blank" in {issue["code"] for issue in report["issues"]}


def test_task_creation_is_not_mistaken_for_a_task_update():
    messages, threads, reference_issues, examples = _fixture_data()
    creation = copy.deepcopy(examples[0])
    assert creation["example_kind"] == "task_creation"
    assert creation["task_source_message_ids"] == [creation["update_message_id"]]
    creation["relation_labels"] = ["completion"]
    report = validate_examples([creation], messages, threads, reference_issues)
    assert "task_creation_labels" in {issue["code"] for issue in report["issues"]}


def test_message_act_can_coexist_with_unrelated_task_pair_relation():
    messages, threads, reference_issues, examples = _fixture_data()
    text = "Could you send the travel plan?"
    messages["SYN-UNRELATED-02"] = dict(messages["SYN-UNRELATED-02"], cleaned_body=text)
    example = copy.deepcopy(examples[7])
    example["message_act_labels"] = ["request"]
    example["update_text"] = text
    example["evidence_spans"] = [
        {"text_field": "update_text", "start_char": 0, "end_char": 9, "supports_label": "request"},
        {"text_field": "update_text", "start_char": 0, "end_char": len(text), "supports_label": "unrelated"},
    ]
    report = validate_examples([example], messages, threads, reference_issues)
    assert report["error_count"] == 0, report["issues"]


def test_unhashable_wrong_types_are_reported_without_crashing():
    messages, threads, reference_issues, examples = _fixture_data()
    invalid = copy.deepcopy(examples[1])
    invalid["example_kind"] = []
    invalid["example_origin"] = {}
    invalid["annotation_status"] = []
    invalid["task_source_message_ids"] = [{}]
    invalid["message_act_labels"] = [[]]
    invalid["relation_labels"] = [["unrelated"]]
    invalid["evidence_spans"] = [{"text_field": [], "start_char": 0, "end_char": 1, "supports_label": []}]

    report = validate_examples([invalid], messages, threads, reference_issues)

    assert report["error_count"] > 0
    assert {"field_type", "example_kind", "example_origin", "annotation_status"} <= {
        issue["code"] for issue in report["issues"]
    }


def test_missing_message_id_uses_source_path_reference_fallback():
    messages = {
        "path:synthetic/no-id/1": {
            "message_id": "", "thread_id": "SYN-NO-ID", "position": 1,
            "source_relative_path": "synthetic/no-id/1", "cleaned_body": "Prepare the notes.",
        },
        "SYN-NO-ID-2": {
            "message_id": "SYN-NO-ID-2", "thread_id": "SYN-NO-ID", "position": 2,
            "source_relative_path": "synthetic/no-id/2", "cleaned_body": "I will prepare the notes.",
        },
    }
    threads = {"SYN-NO-ID": {"synthetic/no-id/1", "synthetic/no-id/2"}}
    example = {
        "schema_version": "1.0", "example_id": "SYN-NO-ID-PAIR", "example_kind": "task_update",
        "example_origin": "synthetic", "annotation_status": "complete", "thread_id": "SYN-NO-ID",
        "task_id": "SYN-NO-ID-TASK", "task_source_message_ids": ["path:synthetic/no-id/1"],
        "task_text": "Prepare the notes.", "task_owner_text": "", "task_deadline_text": "",
        "task_deadline_normalized": None, "update_message_id": "SYN-NO-ID-2",
        "update_text": "I will prepare the notes.", "message_act_labels": ["commitment"],
        "relation_labels": [], "updated_owner_text": "", "updated_deadline_text": "",
        "updated_deadline_normalized": None,
        "evidence_spans": [{"text_field": "update_text", "start_char": 0, "end_char": 6, "supports_label": "commitment"}],
        "ambiguous": False, "ambiguity_reason": "", "annotator_id": "SYNTHETIC_DEMO", "annotation_notes": "",
    }
    report = validate_examples([example], messages, threads)
    assert report["error_count"] == 0, report["issues"]


def test_screening_draft_keeps_task_provenance_blank_until_human_review():
    messages, threads, reference_issues, _ = _fixture_data()
    draft = {
        "schema_version": "1.0", "example_id": "SYN-SCREEN-DRAFT", "example_kind": "screening_draft",
        "example_origin": "synthetic", "annotation_status": "draft", "thread_id": "SYN-NEW",
        "task_id": "", "task_source_message_ids": [], "task_text": "", "task_owner_text": "",
        "task_deadline_text": "", "task_deadline_normalized": None, "update_message_id": "SYN-NEW-01",
        "update_text": "Please prepare the agenda.", "message_act_labels": [], "relation_labels": [],
        "updated_owner_text": "", "updated_deadline_text": "", "updated_deadline_normalized": None,
        "evidence_spans": [], "ambiguous": False, "ambiguity_reason": "", "annotator_id": "", "annotation_notes": "",
    }
    report = validate_examples([draft], messages, threads, reference_issues)
    assert report["error_count"] == 0, report["issues"]

    draft["task_text"] = "Invented unverified task"
    report = validate_examples([draft], messages, threads, reference_issues)
    assert "draft_not_blank" in {issue["code"] for issue in report["issues"]}


def test_reference_thread_membership_and_count_mismatches_are_reported(tmp_path):
    messages_path = tmp_path / "messages.jsonl"
    threads_path = tmp_path / "threads.jsonl"
    messages_path.write_text(json.dumps({
        "message_id": "REF-1", "thread_id": "REF-T1", "position": 1,
        "source_relative_path": "mail-a/inbox/1", "cleaned_body": "Task text.",
    }) + "\n", encoding="utf-8")
    threads_path.write_text(json.dumps({
        "thread_id": "REF-T1", "message_count": 2, "source_relative_paths": ["mail-a/inbox/missing"],
    }) + "\n", encoding="utf-8")

    _, _, issues = load_reference_index(messages_path, threads_path)

    assert {"reference_thread_count_mismatch", "missing_reference_thread_member", "message_missing_from_reference_thread"} <= {
        issue["code"] for issue in issues
    }


@unittest.skipUnless(SMOKE_OUTPUTS_PRESENT, "local smoke thread outputs and candidate CSVs are required")
def test_blank_draft_references_permitted_smoke_thread_without_modifying_inputs(tmp_path):
    root = REPO_ROOT
    smoke_dirs = [
        root / "data/processed/enron_subset_smoke_threads",
        root / "data/processed/enron_subset_interrupt_smoke_threads",
    ]
    candidate_files = [root / "data/annotations/candidates.smoke.csv", root / "data/annotations/candidates.interrupt_smoke.csv"]
    tracked_inputs = candidate_files + [smoke / filename for smoke in smoke_dirs for filename in ("messages.jsonl", "threads.jsonl")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked_inputs}

    for smoke_index, smoke in enumerate(smoke_dirs):
        messages_path = smoke / "messages.jsonl"
        threads_path = smoke / "threads.jsonl"
        by_thread = {}
        for line in messages_path.read_text(encoding="utf-8").splitlines():
            message = json.loads(line)
            by_thread.setdefault(message["thread_id"], []).append(message)
        ordered = next(
            sorted(messages, key=lambda message: message["position"])
            for messages in by_thread.values() if len(messages) >= 2
        )
        source, update = ordered[0], ordered[1]
        draft = {
            "schema_version": "1.0", "example_id": f"DRAFT-SMOKE-{smoke_index}", "example_kind": "task_update",
            "example_origin": "real", "annotation_status": "draft", "thread_id": update["thread_id"],
            "task_id": "UNANNOTATED-DRAFT", "task_source_message_ids": [source["message_id"]],
            "task_text": source["cleaned_body"], "task_owner_text": "", "task_deadline_text": "",
            "task_deadline_normalized": None, "update_message_id": update["message_id"],
            "update_text": update["cleaned_body"], "message_act_labels": [], "relation_labels": [],
            "updated_owner_text": "", "updated_deadline_text": "", "updated_deadline_normalized": None,
            "evidence_spans": [], "ambiguous": False, "ambiguity_reason": "",
            "annotator_id": "", "annotation_notes": "",
        }
        annotations = tmp_path / f"blank_draft_{smoke_index}.jsonl"
        annotations.write_text(json.dumps(draft) + "\n", encoding="utf-8")
        report = validate_file(annotations, messages_path, threads_path)
        assert report["error_count"] == 0, report["issues"]
        assert report["draft_count"] == 1 and report["complete_count"] == 0

    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked_inputs} == before