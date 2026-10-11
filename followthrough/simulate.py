"""Synthetic thread generator with KNOWN gold final state.

Purpose: (1) get thousands of labeled training threads cheaply, (2) unit-test the
tracker. Real-data evaluation (hand-annotated AMI/Enron threads) is still required
for the paper; synthetic-only results must never be reported as the main result.
Later step: paraphrase the templates with an LLM to widen language variety.
"""
import random
from .schema import Message, Item, Thread, ACTIVE, CANCELED, DONE

NAMES = ["Alex", "Priya", "Sam", "Jordan", "Mei", "Carlos"]
TASKS = ["the slides", "the budget sheet", "the introduction draft", "the survey results",
         "the meeting notes", "the lab report"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

REQ = ["{o}, please send {t} by {d}.", "{o}, can you finish {t} by {d}?", "Could you handle {t}, {o}? Due {d}."]
DL = ["Actually, make {t} due {d} instead.", "Change of plan: {t} is now due {d}."]
RE = ["Actually, {n} will handle {t}.", "{n} is taking over {t} now."]
CA = ["Never mind about {t}.", "We don't need {t} anymore."]
CO = ["I sent {t}.", "{t} is done."]
CH = ["Sounds good.", "Thanks everyone.", "Lunch at noon?", "Ok."]


def make_thread(rng: random.Random, tid: str) -> Thread:
    people = rng.sample(NAMES, 3)
    boss, owner, other = people
    n_items = rng.choice([1, 2])
    tasks = rng.sample(TASKS, n_items)
    msgs, items = [], []

    def say(sp, txt):
        msgs.append(Message(len(msgs), sp, txt))
        return len(msgs) - 1

    for i, t in enumerate(tasks):
        d = rng.choice(DAYS)
        m = say(boss, rng.choice(REQ).format(o=owner, t=t, d=d))
        it = Item(i, t, owner, d, ACTIVE, evidence=[m])
        say(rng.choice(people), rng.choice(CH))
        # 0-2 later events; the LAST one decides the final state
        for _ in range(rng.choice([0, 1, 1, 2])):
            ev = rng.choice(["DL", "RE", "CA", "CO"])
            if it.status != ACTIVE:
                break
            if ev == "DL":
                nd = rng.choice([x for x in DAYS if x != it.deadline]); it.deadline = nd
                m = say(boss, rng.choice(DL).format(t=t, d=nd))
            elif ev == "RE":
                nn = rng.choice([p for p in people if p != it.owner]); it.owner = nn
                m = say(boss, rng.choice(RE).format(n=nn, t=t))
            elif ev == "CA":
                it.status = CANCELED; m = say(boss, rng.choice(CA).format(t=t))
            else:
                it.status = DONE; m = say(it.owner, rng.choice(CO).format(t=t.capitalize()))
            it.evidence.append(m)
            say(rng.choice(people), rng.choice(CH))
        items.append(it)
    return Thread(tid, msgs, items)


def make_dataset(n: int, seed: int = 0):
    rng = random.Random(seed)
    return [make_thread(rng, f"syn-{i}") for i in range(n)]
