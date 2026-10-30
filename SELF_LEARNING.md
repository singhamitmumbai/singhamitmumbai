# 24/7 self-learning system

Complete AI can **learn continuously** from verified experience — day and night — without claiming AGI.

> `human_level_agi=false`  
> Unverified model/self predictions are **never** accepted as ground-truth labels.

---

## Enable

```bash
export COMPLETE_AI_CONTINUOUS_LEARNING=1
python -m cognitive_ai learn --daemon --interval 60
```

Or in-product:

```text
start continuous learning
learning status
learning tick
stop continuous learning
```

```python
from cognitive_ai.complete_ai import CompleteAI
ai = CompleteAI(seed_demo=True, enable_llm=False)
ai.start_learning(interval_sec=60)   # background thread
print(ai.learning_status())
# ... later
ai.stop_learning()
```

---

## What it learns (every tick)

| Job | Behavior |
|-----|----------|
| **drain_inbox** | Consume session observations → episodes + preferences + skill outcomes |
| **reinforce_skills** | Reinforce skills with ≥3 verified successes |
| **consolidate** | Lifelong episodic → claims / micro-procedures |
| **mine_patterns** | Frequent routes become *candidate* preferences (not silent facts) |
| **dream** | Replay high-importance episodes |
| **calibrate** | Update metacognition success calibration |

Ticks are **bounded** (default ≤6 jobs) so the loop stays stable 24/7.

---

## Safety / honesty

1. Off unless `COMPLETE_AI_CONTINUOUS_LEARNING=1` or `start_learning()`  
2. Rejects `source in {model, self, unverified}` without `verified=True`  
3. Pattern mining creates **candidates**, not auto-asserted world facts  
4. Durable journal under `data/complete_ai/continuous_learning/`  
5. Snapshot survives process restart  

---

## CLI

```bash
# 3 fast ticks (demo)
python -m cognitive_ai learn --ticks 3 --interval 1

# Real 24/7 daemon
export COMPLETE_AI_CONTINUOUS_LEARNING=1
python -m cognitive_ai learn --daemon --interval 60

# Status / stop
python -m cognitive_ai learn --ask status
python -m cognitive_ai learn --ask stop
```

---

## Verify

```bash
pytest tests/test_continuous_learning_v125.py -q
python -m cognitive_ai learn --ticks 2
```
