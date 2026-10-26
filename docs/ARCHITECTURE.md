# Complete Non-LLM AI — Architecture

This repository implements the engineering blueprint for a **complete non-LLM AI**:
an assistant that understands supported requests, maintains explicit knowledge,
reasons, plans, acts, verifies, and learns **without a pretrained general-purpose
LLM at runtime**.

## Operating process

```
Input → Perception → Language understanding → Working state
      ↔ Knowledge / memory / attention
      → Reasoning → Planning → Execute → Verify → Response
      → Reviewed experience → Learning / evaluation
```

## Major packages

| Package | Responsibility |
|---------|----------------|
| `cognitive_ai/core` | Typed records: Observation, Entity, Claim, Goal, Plan, ActionResult, LearningExample; WorkingState |
| `cognitive_ai/perception` | Input gateway, speech & OCR pipelines (stubs + contracts) |
| `cognitive_ai/nlu` | Intent, slots, Hindi/Hinglish, dialogue state — no LLM |
| `cognitive_ai/memory` | Ontology, semantic/episodic/procedural/preference memory, attention |
| `cognitive_ai/reasoning` | Production rules, arithmetic, competing explanations |
| `cognitive_ai/world` | Explicit transition model T(s′\|s,a) |
| `cognitive_ai/goals` | Goal manager + operator planner |
| `cognitive_ai/execution` | Executor separate from Verifier; OUTCOME_UNKNOWN |
| `cognitive_ai/response` | Template / grammar-based realization + faithfulness check |
| `cognitive_ai/learning_sys` | Fact/alias/skill learning with review gates |
| `cognitive_ai/evaluation` | Task-level harness + oracle failure location |
| `cognitive_ai/capabilities` | Observable capability specifications |
| `cognitive_ai/domains/finderbay` | Invoices, deliveries, materials, tasks |
| `cognitive_ai/system.py` | `CompleteNonLLMAI` orchestration |
| `cognitive_ai/brain.py` | Earlier symbolic cognitive brain (still available) |

## Distinctions preserved everywhere

- **Unknown ≠ false**
- **Predicted ≠ observed**
- **Attempted ≠ confirmed** (`OUTCOME_UNKNOWN`)

## Non-LLM boundary

Allowed: rules, trees/boosting, Bayesian models, CNNs, RNNs/CTC for speech, CRFs,
RL, graph algorithms, search/planning, small n-gram LMs for decoding.

Excluded at runtime: pretrained general-purpose LLMs.

Perception modules ship **auditable stubs**; production acoustic/OCR checkpoints
must declare provenance separately.

## Quick start

```bash
pip install -e ".[dev]"
python -m cognitive_ai finderbay --demo
python -m cognitive_ai eval
pytest -q
```
