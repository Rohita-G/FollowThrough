"""Sequence Classification Models for FollowThrough: Fine-Tuned DistilBERT & TF-IDF Baseline.

Provides clean object-oriented wrappers for:
1. Fine-tuned DistilBERT multi-label sequence classifier (Request & Commitment).
2. TF-IDF + Logistic Regression baseline classifier.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from followthrough_core.long_message_classifier import classify_long_message


class DistilBERTClassifier:
    """Wrapper for fine-tuned DistilBERT request and commitment classification."""

    def __init__(
        self,
        model_path_or_name: Union[str, Path] = "distilbert/distilbert-base-uncased",
        device: Optional[torch.device] = None,
        max_length: int = 128,
    ):
        self.model_path = str(model_path_or_name)
        self.max_length = max_length
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        text: str,
        req_thresh: float = 0.35,
        com_thresh: float = 0.20,
        use_sliding_window: bool = False,
        stride: int = 64,
    ) -> Dict[str, Any]:
        """Classify input message into request and commitment probabilities and predictions.
        
        Args:
            text: Input message text.
            req_thresh: Decision threshold for request.
            com_thresh: Decision threshold for commitment.
            use_sliding_window: If True, uses overlapping windows to process entire text.
            stride: Token stride for sliding window.
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "req_score": 0.0,
                "com_score": 0.0,
                "req_pred": 0,
                "com_pred": 0,
                "token_count": 0,
                "truncated": False,
                "sliding_window": None,
            }

        tokens = self.tokenizer.encode(clean_text, truncation=False)
        token_count = len(tokens)

        if use_sliding_window:
            res = classify_long_message(
                clean_text,
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.device,
                max_window_length=self.max_length,
                stride=stride,
                req_thresh=req_thresh,
                com_thresh=com_thresh,
            )
            return {
                "req_score": res["max_req_score"],
                "com_score": res["max_com_score"],
                "req_pred": res["req_pred"],
                "com_pred": res["com_pred"],
                "token_count": token_count,
                "truncated": False,
                "sliding_window": res,
            }

        truncated = token_count > self.max_length
        enc = self.tokenizer(
            clean_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        req_score = float(probs[0])
        com_score = float(probs[1])

        return {
            "req_score": req_score,
            "com_score": com_score,
            "req_pred": int(req_score >= req_thresh),
            "com_pred": int(com_score >= com_thresh),
            "token_count": token_count,
            "truncated": truncated,
            "sliding_window": None,
        }


class TFIDFBaselineClassifier:
    """Wrapper for TF-IDF + Logistic Regression baseline classification."""

    def __init__(self, model_bundle_path: Union[str, Path]):
        self.model_path = Path(model_bundle_path)
        bundle = joblib.load(self.model_path)
        self.vectorizer = bundle.get("vectorizer")
        
        # Support both {'clf_req', 'clf_com'} and {'classifiers': [req, com]} bundle schemas
        if "clf_req" in bundle and "clf_com" in bundle:
            self.clf_req = bundle["clf_req"]
            self.clf_com = bundle["clf_com"]
        elif "classifiers" in bundle:
            self.clf_req = bundle["classifiers"][0]
            self.clf_com = bundle["classifiers"][1]
        else:
            raise ValueError(f"Unrecognized bundle format in {self.model_path}")

        self.thresholds = bundle.get("thresholds", {"request": 0.45, "commitment": 0.65})

    def predict(
        self,
        text: str,
        req_thresh: Optional[float] = None,
        com_thresh: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Classify input text using TF-IDF and logistic regression."""
        clean_text = text.strip()
        if not clean_text:
            return {
                "req_score": 0.0,
                "com_score": 0.0,
                "req_pred": 0,
                "com_pred": 0,
            }

        r_thresh = req_thresh if req_thresh is not None else self.thresholds.get("request", 0.45)
        c_thresh = com_thresh if com_thresh is not None else self.thresholds.get("commitment", 0.65)

        feat = self.vectorizer.transform([clean_text])
        req_score = float(self.clf_req.predict_proba(feat)[0][1])
        com_score = float(self.clf_com.predict_proba(feat)[0][1])

        return {
            "req_score": req_score,
            "com_score": com_score,
            "req_pred": int(req_score >= r_thresh),
            "com_pred": int(com_score >= c_thresh),
        }


def load_distilbert_classifier(
    model_path_or_name: Union[str, Path],
    device: Optional[torch.device] = None,
) -> DistilBERTClassifier:
    """Helper to load a DistilBERT classifier instance."""
    return DistilBERTClassifier(model_path_or_name=model_path_or_name, device=device)


def load_tfidf_classifier(model_bundle_path: Union[str, Path]) -> TFIDFBaselineClassifier:
    """Helper to load a TF-IDF baseline classifier instance."""
    return TFIDFBaselineClassifier(model_bundle_path=model_bundle_path)
