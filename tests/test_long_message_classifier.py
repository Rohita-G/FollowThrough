"""Tests for sliding-window tokenization and classification over long messages without Calendar remnants."""

from pathlib import Path
from unittest.mock import MagicMock
import torch
from transformers import AutoTokenizer

from followthrough_core.long_message_classifier import classify_long_message
from followthrough_core.task_extractor import extract_candidate_tasks


def test_long_message_sliding_window_offsets_and_coverage():
    # Use standard uncased tokenizer or fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    device = torch.device("cpu")

    # Mock model returning dummy logits for 2 classes: [req_logit, com_logit]
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[1.5, -0.5]])
    mock_model.return_value = mock_output

    # Construct a synthetic message longer than 128 tokens
    padding_sentences = [
        f"This is background paragraph number {i} discussing general operations and administrative procedures."
        for i in range(12)
    ]
    action_sentence = "Please submit the finalized financial risk report by Friday afternoon."
    full_message = " ".join(padding_sentences) + " " + action_sentence

    tokens = tokenizer.encode(full_message, truncation=False)
    assert len(tokens) > 130, f"Expected >130 tokens, got {len(tokens)}"

    # Run sliding window classifier
    res = classify_long_message(
        full_message,
        tokenizer=tokenizer,
        model=mock_model,
        device=device,
        max_window_length=128,
        stride=64,
        req_thresh=0.5,
        com_thresh=0.5,
    )

    assert res["num_windows"] >= 2
    assert res["total_tokens"] == len(tokenizer.encode(full_message, add_special_tokens=False))

    # Verify per-window special tokens, offsets, and original text slicing
    for w in res["windows"]:
        assert w["token_count"] <= 128
        assert full_message[w["char_start"]:w["char_end"]] == w["source_text"]
        assert 0.0 <= w["req_score"] <= 1.0
        assert 0.0 <= w["com_score"] <= 1.0

    # Ensure last window catches the action sentence at the end
    last_window = res["windows"][-1]
    assert "financial risk report" in last_window["source_text"]

    # Verify that task extraction extracts from the FULL message independent of 128 truncation
    candidates = extract_candidate_tasks(full_message)
    assert len(candidates) >= 1
    found_task = any("financial risk report" in c["action_description"] for c in candidates)
    assert found_task is True
    cand = candidates[0]
    assert full_message[cand["start_offset"]:cand["end_offset"]] == cand["evidence_text"]


def test_empty_message_handling():
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    mock_model = MagicMock()
    res = classify_long_message("", tokenizer=tokenizer, model=mock_model, device=torch.device("cpu"))
    assert res["num_windows"] == 0
    assert res["total_tokens"] == 0
    assert res["max_req_score"] == 0.0


if __name__ == "__main__":
    test_long_message_sliding_window_offsets_and_coverage()
    test_empty_message_handling()
    print("All sliding window long message tests passed successfully!")
