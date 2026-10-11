from followthrough.simulate import make_dataset
from followthrough.baselines import MessageLevel, ThreadAware
from followthrough.metrics import evaluate

def test_thread_aware_beats_message_level():
    d = make_dataset(200, seed=3)
    a, b = evaluate(d, MessageLevel()), evaluate(d, ThreadAware())
    assert b["stale_task_rate"] < a["stale_task_rate"]
    assert b["active_f1"] > a["active_f1"]
