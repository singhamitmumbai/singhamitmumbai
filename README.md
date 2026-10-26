# Complete Non-LLM AI (Finderbay + Symbolic Cognitive OS)

A runnable **complete non-LLM AI** (v0.6) built from the engineering blueprint:

perception → structured understanding → knowledge/memory → reasoning → planning →
learning → action → verification — with a shared typed working state and **no
pretrained general-purpose LLM at runtime**.

### What v0.6 adds toward “general” capability

| Area | Upgrade |
|------|---------|
| Open-domain chat | Curated common-sense KB + FAQ + analogies (not ChatGPT-scale) |
| Speech | **Vosk** offline ASR (`models/vosk-model-small-en-us-0.15`) |
| OCR | **Tesseract 5** production OCR (`eng+hin`) |
| Generality | `GeneralNonLLMAI` router: Finderbay tasks ↔ open-domain chat |
| Transfer | Skill-transfer helper across projects |

**Honest limits (still):** not human-level AGI; open-domain coverage is finite/explicit; not unrestricted LLM chat.

## Quick start

```bash
pip install -e ".[dev,perception]"
# system OCR: tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin

# General AI (Finderbay + open-domain)
python -m cognitive_ai general --demo
python -m cognitive_ai general --ask "Can penguins fly?"

# Finderbay-only
python -m cognitive_ai finderbay --demo

# Chat UI (general router by default)
python -m cognitive_ai chat --domain general
# open http://127.0.0.1:8765/

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
