"""Overlapping sliding-window inference for long messages with special token handling."""

from typing import Any, Dict, List, Tuple
import torch


def classify_long_message(
    text: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_window_length: int = 128,
    stride: int = 64,
    req_thresh: float = 0.5,
    com_thresh: float = 0.5,
) -> Dict[str, Any]:
    """Perform sliding window classification over long messages.

    DistilBERT requires [CLS] (id 101) and [SEP] (id 102).
    Maximum content tokens per window is max_window_length - 2 = 126.
    Overlapping windows ensure complete message coverage without boundary loss.
    """
    clean_text = text.strip()
    if not clean_text:
        return {
            "mode": "full_message_sliding_window",
            "total_tokens": 0,
            "num_windows": 0,
            "windows": [],
            "max_req_score": 0.0,
            "max_com_score": 0.0,
            "req_pred": 0,
            "com_pred": 0,
            "aggregation_label": "Experimental heuristic (not calibrated confidence)",
        }

    # Tokenize without truncation and obtain character offsets
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offset_mapping = enc["offset_mapping"]
    total_tokens = len(input_ids)

    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102
    max_content_tokens = max_window_length - 2

    windows: List[Dict[str, Any]] = []

    start_idx = 0
    window_idx = 0

    while start_idx < total_tokens:
        end_idx = min(start_idx + max_content_tokens, total_tokens)
        chunk_ids = input_ids[start_idx:end_idx]
        
        # Character offsets in original text
        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx - 1][1]
        window_text = text[start_char:end_char]

        # Assemble input with special tokens
        window_input_ids = [cls_id] + chunk_ids + [sep_id]
        attention_mask = [1] * len(window_input_ids)

        # Tensor inputs
        inp_tensor = torch.tensor([window_input_ids], device=device)
        mask_tensor = torch.tensor([attention_mask], device=device)

        with torch.no_grad():
            logits = model(input_ids=inp_tensor, attention_mask=mask_tensor).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        req_score = float(probs[0])
        com_score = float(probs[1])

        windows.append({
            "window_index": window_idx,
            "token_start": start_idx,
            "token_end": end_idx,
            "char_start": start_char,
            "char_end": end_char,
            "source_text": window_text,
            "token_count": len(window_input_ids),
            "req_score": req_score,
            "com_score": com_score,
            "req_pred": int(req_score >= req_thresh),
            "com_pred": int(com_score >= com_thresh),
        })

        if end_idx == total_tokens:
            break

        start_idx += stride
        window_idx += 1

    max_req = max((w["req_score"] for w in windows), default=0.0)
    max_com = max((w["com_score"] for w in windows), default=0.0)

    return {
        "mode": "full_message_sliding_window",
        "total_tokens": total_tokens,
        "num_windows": len(windows),
        "windows": windows,
        "max_req_score": max_req,
        "max_com_score": max_com,
        "req_pred": int(max_req >= req_thresh),
        "com_pred": int(max_com >= com_thresh),
        "aggregation_label": "Experimental heuristic (Max score across overlapping windows; not calibrated confidence)",
    }
