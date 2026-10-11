"""python run_demo.py  ->  compares the two reference systems on 500 synthetic threads
and prints one worked example (the 'slides' scenario from the project pitch)."""
from followthrough.simulate import make_dataset
from followthrough.baselines import MessageLevel, ThreadAware
from followthrough.metrics import evaluate

data = make_dataset(500, seed=1)
print(f"{'system':28s}" + "".join(f"{k:>17s}" for k in
      ["active_f1", "stale_task_rate", "owner_acc", "deadline_acc", "status_acc"]))
for sys_ in (MessageLevel(), ThreadAware()):
    r = evaluate(data, sys_)
    print(f"{sys_.name:28s}" + "".join(f"{r[k]:17.3f}" for k in
          ["active_f1", "stale_task_rate", "owner_acc", "deadline_acc", "status_acc"]))

th = next(t for t in data if any(i.status != "active" for i in t.gold_items))
print("\nExample thread:")
for m in th.messages: print(f"  [{m.idx}] {m.speaker}: {m.text}")
for s in (MessageLevel(), ThreadAware()):
    print(f"\n{s.name}:")
    for i in s.run(th): print(f"  {i.description!r:28} owner={i.owner} due={i.deadline} status={i.status}")
print("\nGOLD:")
for i in th.gold_items: print(f"  {i.description!r:28} owner={i.owner} due={i.deadline} status={i.status}")
