"""Core data structures shared by every module.

Design note: tasks and decisions share ONE state model (`Item`). That is what lets
the DecisionLedger extension (Project 3) reuse the same tracker later: a decision is
just an Item with kind="decision" and events PROPOSE / AGREE / REVISE instead of
REQUEST / COMMIT / CANCEL.
"""
from dataclasses import dataclass, field
from typing import List, Optional

# Event labels a message can express about an item (what the 2nd model must predict).
EVENTS = ["NONE", "REQUEST", "COMMIT", "DEADLINE_CHANGE", "REASSIGN", "CANCEL", "COMPLETE"]
# Reserved for the DecisionLedger extension:
DECISION_EVENTS = ["PROPOSE", "AGREE", "REVISE", "REJECT"]

# Item status values.
ACTIVE, CANCELED, DONE = "active", "canceled", "done"


@dataclass
class Message:
    idx: int
    speaker: str
    text: str


@dataclass
class Item:
    item_id: int
    description: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    status: str = ACTIVE
    kind: str = "task"                  # "task" now, "decision" in Project 3
    evidence: List[int] = field(default_factory=list)  # message idx values


@dataclass
class Thread:
    thread_id: str
    messages: List[Message]
    gold_items: List[Item] = field(default_factory=list)   # final gold state
