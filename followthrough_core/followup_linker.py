"""Semantic Follow-up Linking and Conservative Update Detection for FollowThrough.

Combines:
1. Dense bi-encoder semantic similarity via sentence-transformers/all-MiniLM-L6-v2
   to rank stored candidate tasks against incoming follow-up messages.
2. Transparent rule-based update detector to catch explicit deadline revisions,
   reassignments, cancellations, and reported completions with exact evidence.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Rule patterns detecting explicit deadline revisions (e.g., postponed to Friday)
DEADLINE_REVISION_PATTERNS = [
    re.compile(
        r"\b(?:push(?:ed)?|mov(?:ed)?|postpon(?:ed)?|reschedul(?:ed)?|delay(?:ed)?)(?:\s+[\w\s]{0,40}?)?\s+(?:to|until|for)\s+(next\s+[A-Za-z]+|[A-Z][a-z]+(?:\s+\d{1,2})?|\d{1,2}/\d{1,2}|tomorrow|today)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:new\s+deadline|due\s+date\s+is\s+now|extended\s+to)\s+(next\s+[A-Za-z]+|[A-Z][a-z]+(?:\s+\d{1,2})?|\d{1,2}/\d{1,2}|tomorrow|today)\b",
        re.I,
    ),
]

# Rule patterns detecting task reassignment / owner changes
OWNER_REVISION_PATTERNS = [
    re.compile(r"\b(?:reassigned\s+to|handing\s+(?:this\s+)?over\s+to|giving\s+this\s+to)\s+([A-Z][a-z]+)\b", re.I),
    re.compile(r"\b([A-Z][a-z]+)\s+will\s+(?:take\s+over|lead|handle\s+this)\b", re.I),
]

# Rule patterns detecting task cancellation
CANCELLATION_PATTERNS = [
    re.compile(r"\b(cancel(?:led)?|calling\s+off|no\s+longer\s+needed|please\s+disregard|drop\s+this)\b", re.I),
]

# Rule patterns detecting reported completion
COMPLETION_PATTERNS = [
    re.compile(
        r"\b(i\s+have\s+(?:submitted|completed|finished|sent|filed)|task\s+is\s+(?:done|complete)|finished\s+the\s+review|all\s+done)\b",
        re.I,
    ),
]

# Conversational pleasantries explicitly excluded from completion detection
COURTESY_ONLY = re.compile(r"^(?:thanks|thank\s+you|thx|great|appreciated|looks\s+good)[\s.!,]*$", re.I)


class FollowupLinker:
    """Computes semantic similarity embeddings and manages task ranking."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model on first usage."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compute_similarity(self, query: str, candidate_texts: List[str]) -> np.ndarray:
        """Compute cosine similarity between query text and a list of stored candidate descriptions."""
        if not candidate_texts:
            return np.array([])
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        c_embs = self.model.encode(candidate_texts, convert_to_numpy=True, normalize_embeddings=True)
        sims = np.dot(c_embs, q_emb.T).squeeze(axis=1)
        return sims

    def rank_candidates(
        self,
        followup_text: str,
        saved_tasks: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Rank stored tasks against follow-up message using semantic similarity scores.
        
        Returns top_k candidates with similarity scores (0.0 to 1.0).
        Note: High similarity indicates semantic proximity, but is not proof of a relationship.
        """
        if not saved_tasks:
            return []

        descriptions = [t["action_description"] for t in saved_tasks]
        sims = self.compute_similarity(followup_text, descriptions)

        ranked = []
        for task, score in zip(saved_tasks, sims):
            ranked.append({
                "task": task,
                "similarity_score": round(float(score), 4),
            })

        ranked.sort(key=lambda x: x["similarity_score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def detect_updates(followup_text: str) -> List[Dict[str, Any]]:
        """Conservatively detect proposed updates from follow-up messages with exact evidence.

        Rejects pure courtesy messages (e.g. 'Thanks!') so they are never marked as completed.
        Returns candidate revisions for user review before database commit.
        """
        proposals: List[Dict[str, Any]] = []
        clean = followup_text.strip()

        # Reject pure courtesies
        if COURTESY_ONLY.match(clean):
            return proposals

        # 1. Deadline Revision Detection
        for pat in DEADLINE_REVISION_PATTERNS:
            m = pat.search(clean)
            if m:
                proposals.append({
                    "update_type": "deadline_revised",
                    "new_value": m.group(1),
                    "evidence_text": m.group(0),
                    "confidence": "rule_detected",
                    "description": f"Proposed deadline revision to: '{m.group(1)}'",
                })
                break

        # 2. Owner Reassignment Detection
        for pat in OWNER_REVISION_PATTERNS:
            m = pat.search(clean)
            if m:
                proposals.append({
                    "update_type": "owner_revised",
                    "new_value": m.group(1),
                    "evidence_text": m.group(0),
                    "confidence": "rule_detected",
                    "description": f"Proposed owner reassignment to: '{m.group(1)}'",
                })
                break

        # 3. Task Cancellation Detection
        for pat in CANCELLATION_PATTERNS:
            m = pat.search(clean)
            if m:
                proposals.append({
                    "update_type": "cancelled",
                    "new_value": "cancelled",
                    "evidence_text": m.group(0),
                    "confidence": "rule_detected",
                    "description": "Proposed task cancellation",
                })
                break

        # 4. Reported Completion Detection
        for pat in COMPLETION_PATTERNS:
            m = pat.search(clean)
            if m:
                proposals.append({
                    "update_type": "completed",
                    "new_value": "completed",
                    "evidence_text": m.group(0),
                    "confidence": "rule_detected",
                    "description": "Proposed task completion",
                })
                break

        return proposals
