"""Unit tests for DistilBERTClassifier and TFIDFBaselineClassifier wrappers."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch

from followthrough_core.classifier import DistilBERTClassifier, TFIDFBaselineClassifier


def test_tfidf_baseline_classifier_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "test_tfidf.joblib"

        # Fit a miniature TF-IDF vectorizer and classifiers
        train_texts = [
            "Please send the report by tomorrow.",
            "I will submit the spreadsheet soon.",
            "The weather is very nice today.",
            "Can you review the attached draft?",
        ]
        y_req = np.array([1, 0, 0, 1])
        y_com = np.array([0, 1, 0, 0])

        vec = TfidfVectorizer()
        X = vec.fit_transform(train_texts)

        clf_r = LogisticRegression()
        clf_r.fit(X, y_req)

        clf_c = LogisticRegression()
        clf_c.fit(X, y_com)

        bundle = {
            "vectorizer": vec,
            "clf_req": clf_r,
            "clf_com": clf_c,
            "thresholds": {"request": 0.45, "commitment": 0.65},
        }
        joblib.dump(bundle, bundle_path)

        # Test TFIDFBaselineClassifier
        classifier = TFIDFBaselineClassifier(bundle_path)
        pred = classifier.predict("Please send the update.")

        assert "req_score" in pred
        assert "com_score" in pred
        assert "req_pred" in pred
        assert "com_pred" in pred
        assert 0.0 <= pred["req_score"] <= 1.0
        assert 0.0 <= pred["com_score"] <= 1.0

        # Empty string test
        empty_pred = classifier.predict("")
        assert empty_pred["req_pred"] == 0
        assert empty_pred["com_pred"] == 0


def test_distilbert_classifier_wrapper():
    # Test initialization with mock model & tokenizer
    classifier = object.__new__(DistilBERTClassifier)
    classifier.max_length = 128
    classifier.device = torch.device("cpu")

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [101, 2000, 102]
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2000, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    classifier.tokenizer = mock_tokenizer

    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[2.0, -1.0]])
    mock_model.return_value = mock_output
    classifier.model = mock_model

    result = classifier.predict("Please check this file.", req_thresh=0.5, com_thresh=0.5)
    assert result["req_pred"] == 1
    assert result["com_pred"] == 0
    assert result["req_score"] > 0.8
    assert result["token_count"] == 3


if __name__ == "__main__":
    test_tfidf_baseline_classifier_predict()
    test_distilbert_classifier_wrapper()
    print("All classifier wrapper tests passed successfully!")
