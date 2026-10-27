# Complete Non-LLM AI (Finderbay + Symbolic Cognitive OS)

A runnable **Complete AI** (v1.3.0) — full stack built from the engineering blueprint:

perception → structured understanding → knowledge/memory → reasoning → planning →
learning → action → verification — with a shared typed working state and **no
pretrained general-purpose LLM at runtime**.

### Complete AI (v1.3) — product entrypoint

```bash
python -m cognitive_ai complete-ai --demo
python -m cognitive_ai complete-ai --ask "Who are you?"
# Optional ChatGPT-class general chat:
#   export COMPLETE_AI_ENABLE_LLM=1 OPENAI_API_KEY=sk-...
#   python -m cognitive_ai complete-ai --mode llm --ask "Write a short poem about Mars"
```

`human_level_agi` stays **false**. LLM chat is **off by default**; when enabled, answers report `llm_at_runtime=true`.

### Release zip

```bash
python scripts/package_release.py
# → dist/Complete_Non_LLM_AI_v1.3.0.zip
```

See `INSTALL.md` for install steps after unzipping.

### What v1.2 adds — Computer vision (YOLO_Projects)

Vendored from [Nawaf-Rayhan585/YOLO_Projects](https://github.com/Nawaf-Rayhan585/YOLO_Projects) (MIT):

| Piece | Role |
|------|------|
| `cognitive_ai/vision/` | VisionSystem + YOLO engine (Ultralytics or stub) + task adapters |
| `third_party/YOLO_Projects/` | Source projects (vehicles, people, fire, PPE, document scan, …) |
| NLU / planner / executor | Intents: detect objects/vehicles/people/fire, scan document, vision status |
| CLI | `python -m cognitive_ai vision --task vision_status` · `--image path.jpg` |

CNN detectors are specialist **perception** (blueprint §5C) — not a general LLM.

### What v1.1 adds — SelfBrain agent + coding system

| Piece | Role |
|------|------|
| `SelfBrainAgent` | Autonomous queue: take Finderbay tasks → plan → act → verify → retry |
| `FinderbayAgentSystem` | Bridge so Finderbay AI assigns work the SelfBrain executes |
| `CodingSystem` | Advanced **non-LLM** coding: AST analysis, symbol index, templates, edits, pytest |
| CLI | `python -m cognitive_ai agent --demo` · `python -m cognitive_ai code --ask "..."` |

### What v1.0 completes (blueprint alignment)

| Blueprint theme | Implementation |
|------|---------|
| Shared operating cycle | Inspectable `cycle` trace on every request (perceive→…→respond) |
| Uncertainty routing | Separate kinds → clarify / compare / refresh / abstain / qualify |
| Conflict detection | Conflicting claims & invoice↔payment mismatches exposed |
| Case-based reasoning | Reuse verified episodes as plan hints |
| Improvement process | Failure → reviewed candidate → release (no self-labeling) |
| Program induction | Learn operator sequences from reviewed examples |
| Blueprint audit | `python -m cognitive_ai blueprint` component inventory |

Also includes v0.9 question understanding, v0.8 alerts/POs/API, speech/OCR, 2000+ CS KB.

**Honest limits (still):** not human-level AGI; open-domain coverage is finite/explicit; not unrestricted LLM chat.

## Quick start

```bash
pip install -e ".[dev,perception]"
# optional real YOLO CNN inference:
# pip install -e ".[vision]"
# system OCR: tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin

# General AI (Finderbay + open-domain)
python -m cognitive_ai general --demo
python -m cognitive_ai general --ask "Can penguins fly?"

# Finderbay-only
python -m cognitive_ai finderbay --demo

# Computer vision (stub without ultralytics; real CNN with [vision] extra)
python -m cognitive_ai vision --task vision_status
python -m cognitive_ai finderbay --ask "Count people in image ./crowd_people.jpg"
python -m cognitive_ai finderbay --ask "Detect vehicles in image ./traffic_cars.jpg"
python -m cognitive_ai vision --image ./photo.jpg --task detect_objects

# Chat UI (general router by default)
python -m cognitive_ai chat --domain general
# open http://127.0.0.1:8765/

python -m cognitive_ai eval
python -m cognitive_ai bench
python -m cognitive_ai blueprint
python -m cognitive_ai api --port 8787
# GET /health /status /alerts /reports?kind=invoices  POST /ask
pytest -q
```

Speech models (optional, for live ASR): place Vosk packs under `models/`
(`vosk-model-small-en-us-0.15`, `vosk-model-small-hi-0.22`, `vosk-model-small-en-in-0.4`).
They are gitignored; the model registry reports install status.

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

Speech uses **Vosk** (offline ASR; EN pack vendored, HI/EN-IN via registry download).
OCR uses **Tesseract** (+ layout/table heuristics). No pretrained general-purpose
LLM is loaded at runtime. Acoustic/OCR checkpoints should still be inventoried
for LLM contamination if your policy requires “no LLM anywhere in the lifecycle.”
