"""Two rule-based reference systems. These are NOT the final models; they define the
comparison structure and the interface that the fine-tuned transformers must fill:

  A. MessageLevel : extracts a task from every request sentence, never updates it.
  B. ThreadAware  : links later messages to open tasks and applies the event.

Replace the regex "event detector" in B with (1) a fine-tuned event classifier and
(2) a cross-encoder linker, keeping the same state-update logic.
"""
import re
from typing import List
from .schema import Item, Thread, ACTIVE, CANCELED, DONE
from .simulate import TASKS, DAYS

REQ_RE = re.compile(r"(\w+), (?:please|can you)|Could you handle .*?, (\w+)")
DAY_RE = re.compile("|".join(DAYS))


def _task_of(text: str):
    for t in TASKS:
        if t in text.lower():
            return t
    return None


def _owner_of(text: str):
    m = REQ_RE.search(text)
    return (m.group(1) or m.group(2)) if m else None


class MessageLevel:
    name = "A: message-level"

    def run(self, th: Thread) -> List[Item]:
        items = []
        for m in th.messages:
            if _owner_of(m.text) and _task_of(m.text):
                d = DAY_RE.search(m.text)
                items.append(Item(len(items), _task_of(m.text), _owner_of(m.text),
                                  d.group(0) if d else None, ACTIVE, evidence=[m.idx]))
        return items


class ThreadAware:
    name = "B: thread-aware (rules)"

    def run(self, th: Thread) -> List[Item]:
        items: List[Item] = []
        for m in th.messages:
            low, task = m.text.lower(), _task_of(m.text)
            open_item = next((i for i in items if i.description == task), None)
            if _owner_of(m.text) and task and open_item is None:
                d = DAY_RE.search(m.text)
                items.append(Item(len(items), task, _owner_of(m.text),
                                  d.group(0) if d else None, ACTIVE, evidence=[m.idx]))
            elif open_item:
                open_item.evidence.append(m.idx)
                if "never mind" in low or "don't need" in low:
                    open_item.status = CANCELED
                elif "i sent" in low or "is done" in low:
                    open_item.status = DONE
                elif "will handle" in low or "taking over" in low:
                    open_item.owner = next(x for x in m.text.replace(",", "").split() if x in
                                           {"Alex","Priya","Sam","Jordan","Mei","Carlos"})
                elif DAY_RE.search(m.text):
                    open_item.deadline = DAY_RE.search(m.text).group(0)
        return items
