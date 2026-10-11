"""Evaluation. Three+ metrics are required by the course; these are the core ones.

stale_task_rate  = share of tasks the system shows as ACTIVE that are NOT active in gold
                   (canceled/done) or are shown under the wrong owner.  <- headline metric
active_f1        = P/R/F1 of the active task list, matching on (description, owner)
field_accuracy   = owner / deadline / status accuracy on matched tasks
"""
from typing import Dict, List, Tuple
from .schema import Thread, Item, ACTIVE


def _active(items: List[Item]):
    return {(i.description, i.owner, i.deadline) for i in items if i.status == ACTIVE}


def evaluate(threads: List[Thread], system) -> Dict[str, float]:
    tp = fp = fn = stale = shown = 0
    f_ok = {"owner": 0, "deadline": 0, "status": 0}; f_n = 0
    for th in threads:
        pred = system.run(th)
        g_act, p_act = _active(th.gold_items), _active(pred)
        tp += len(g_act & p_act); fp += len(p_act - g_act); fn += len(g_act - p_act)
        gold_by_desc = {i.description: i for i in th.gold_items}
        for p in pred:
            if p.status != ACTIVE:
                continue
            shown += 1
            g = gold_by_desc.get(p.description)
            if g is None or g.status != ACTIVE or g.owner != p.owner:
                stale += 1
        for p in pred:
            g = gold_by_desc.get(p.description)
            if g:
                f_n += 1
                f_ok["owner"] += p.owner == g.owner
                f_ok["deadline"] += p.deadline == g.deadline
                f_ok["status"] += p.status == g.status
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    out = {"active_precision": P, "active_recall": R,
           "active_f1": 2 * P * R / (P + R) if P + R else 0.0,
           "stale_task_rate": stale / shown if shown else 0.0}
    for k, v in f_ok.items():
        out[f"{k}_acc"] = v / f_n if f_n else 0.0
    return out
