# Complete Non-LLM AI (Finderbay + Symbolic Cognitive OS)

A runnable **complete non-LLM AI** built from the engineering blueprint:

perception → structured understanding → knowledge/memory → reasoning → planning →
learning → action → verification — with a shared typed working state and **no
pretrained general-purpose LLM at runtime**.

Non-LLM does **not** mean non-neural: specialist speech/OCR models may be plugged
in. This repo ships measurable stubs plus full symbolic / algorithmic cognition
and the **Finderbay** domain (invoices, deliveries, materials, tasks).

## Quick start

```bash
pip install -e ".[dev]"

# Full Finderbay assistant demo (blueprint example)
python -m cognitive_ai finderbay --demo
# or
python scripts/demo_finderbay.py

# One-shot
python -m cognitive_ai finderbay --ask "Show unpaid invoices above ₹1 lakh for Project A, excluding disputed ones."

# Independent evaluation suite
python -m cognitive_ai eval
pytest -q
```

## What “complete” means here

Declared capabilities for a business assistant that:

- understands **supported** requests (English / Hinglish)
- maintains **explicit** knowledge with evidence status and validity periods
- reasons with rules, arithmetic and competing explanations
- plans with typed operators
- executes with durable attempt IDs and verifies effects
- generates language from structured answer plans
- learns only from **reviewed** examples (never self-labelled predictions)

See `docs/ARCHITECTURE.md` and `cognitive_ai/capabilities/spec.py`.

## Example (blueprint §20)

```text
USER: From this week's delivery records, find the items still pending for Project A
      and prepare a follow-up. Do not send it.

AI:   Found 1 confirmed pending delivery item(s). 1 additional item(s) have
      unresolved receipt status and were kept separate from confirmed pending.
      A follow-up draft was prepared and was not sent.
```

Distinctions enforced throughout:

| Distinction | Meaning |
|-------------|---------|
| Unknown ≠ false | Missing receipt ≠ confirmed pending |
| Predicted ≠ observed | Forecasts are not evidence |
| Attempted ≠ confirmed | `OUTCOME_UNKNOWN` until reconciled |

## Architecture (logical components)

```
Text/audio/docs → Perception → Typed working state ↔ Knowledge/memory
                              → Reasoning/planning → Execute/verify → Answer
                              → Reviewed experience → Train/eval → improve modules
```

| Module | Role |
|--------|------|
| `core/` | Observation, Entity, Claim, Goal, Plan, ActionResult, WorkingState |
| `nlu/` | Intent, slots, negation/exclusion/comparison, dialogue tracker |
| `memory/` | Ontology, multi-memory store, attention scheduler |
| `reasoning/` | Invoice/delivery rules, competing explanations |
| `world/` | Explicit stock/payment transitions |
| `goals/` | Goal manager + operator planner |
| `execution/` | Executor ⊥ Verifier |
| `response/` | Non-LLM realization + faithfulness check |
| `perception/` | Speech/OCR pipeline contracts |
| `learning_sys/` | Facts, aliases, candidate skills, continual cycle |
| `evaluation/` | Task harness + oracle failure location |
| `domains/finderbay/` | Seed data, eval cases, templates |
| `system.py` | `CompleteNonLLMAI` |
| `brain.py` / `os/` | Earlier symbolic cognitive brain + Cognitive OS (still available) |

## Legacy symbolic brain / Cognitive OS

```bash
python -m cognitive_ai --demo
python -m cognitive_ai os --os-demo
python -m cognitive_ai chat   # http://127.0.0.1:8765/
```

## Dependencies

Runtime: **Python 3.10+ stdlib only**. Dev: `pytest`. No ML/LLM packages required.

## Provenance note

Speech and OCR modules expose full pipeline stages and metrics hooks. Bundled
implementations are stubs/oracles for offline testing. Production checkpoints
must be inventoried for LLM contamination if your policy requires “no LLM
anywhere in the lifecycle.”
