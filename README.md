# Complete Non-LLM AI (Finderbay + Symbolic Cognitive OS)

**Complete AI** (v1.18.0) — ChatGPT-level chat + **integrated proto-AGI gap stack** with multi-goal autonomy, multimodal arbitration, lifelong persist, open-world chat grounding, and pillar scorecards. Still `human_level_agi=false`:

perception → structured understanding → knowledge/memory → reasoning → planning →
learning → action → verification — with a shared typed working state and **no
pretrained general-purpose LLM required at runtime** (optional LLM via API key).

### Complete AI / Finderbay Assistant / JARVIS — start here

```bash
# JARVIS non-LLM controller (tool contracts · jobs · provenance · shortage skill)
python -m cognitive_ai jarvis --demo
python -m cognitive_ai jarvis --eval
python -m cognitive_ai jarvis --ask "Prepare tomorrow's material shortage report for Site-7"

# Finderbay Assistant (voice · memory · vision · planning · computer · learning · JARVIS)
python -m cognitive_ai assistant --demo
python -m cognitive_ai assistant --ask "remember that my warehouse is Central Store"
python -m cognitive_ai assistant --ask "plan Show unpaid invoices for Project A"

# ChatGPT-class general chat + local tools (auto-on when key is set)
export OPENAI_API_KEY=sk-...
python -m cognitive_ai complete-ai --ask "Write a short poem about Mars"
python -m cognitive_ai diagnostics
python -m cognitive_ai completeness

# Enable local computer control (allowlisted; off by default)
export COMPLETE_AI_COMPUTER_CONTROL=1
python -m cognitive_ai assistant --ask "computer open_url https://example.com"

# API + Chat UI
python -m cognitive_ai api --mode assistant --port 8787
python -m cognitive_ai chat --domain complete
```

Product: **Complete AI** / **Finderbay Assistant** / **JARVIS tools**. `human_level_agi=false`.  
Computer control is **gated** (no keylogging); enable explicitly. LLM is optional.

### What v1.18 adds — Persist, multi-goal, arbitrate, scorecard

| Pillar | Upgrade |
|------|------|
| General learning | Session outcomes feed `GeneralLearner` when proto stack is active |
| Open-world | Chat uses expanded world hub before fallback |
| Deep transfer | New `lab` + `logistics` domain schemas |
| Long-horizon | `enqueue_goal` / `run_queue` multi-goal autonomy |
| Multimodality | Session buffer + `arbitrate()` winner selection |
| Lifelong memory | `persist()` / `load()` continuity across restarts |
| Metacognition | Calibration retained from v1.17 |
| Alignment/evals | Pillar **scorecard** in `run_agi_grade_eval` |

```bash
python -m cognitive_ai agi-gap --eval
pytest tests/test_agi_gap_v118.py -q
```

### What v1.17 adds — Product integration + AGI-grade eval suite

| Piece | Role |
|------|------|
| `agi_gap/eval_suite.py` | Cross-pillar AGI-grade-flavored eval (`run_agi_grade_eval`) |
| SmartRouter `proto_agi` | Routes multi-step / transfer / open-world asks to gap stack |
| CompleteAI `proto_stack` | Lazy stack + `bind_backends` for real chat/web actuators |
| Assistant route | `proto_agi` handled before general Complete AI |
| Metacognition | Outcome calibration rates per strategy |

```bash
python -m cognitive_ai agi-gap --eval
python -m cognitive_ai complete-ai --ask "multi-step research photosynthesis then verify"
python -m cognitive_ai assistant --ask "multi-step research DNA then verify"
pytest tests/test_agi_gap_v117.py -q
```

### What v1.16 adds — Deepened proto-AGI gap stack (still not AGI)

| Pillar | Upgrade |
|------|------|
| General learning | `general_learning.py` — curriculum, replay, verified multi-source updates |
| Open-world | Mega world facts, hypotheses, relation hints, probe generator |
| Deep transfer | `compose_transfer` multi-hop + structural overlap |
| Long-horizon | Failure→replan recovery subgoals + verify |
| Multimodality | Confidence-weighted `fuse()` across channels |
| Lifelong memory | Importance scoring + `dream()` spaced replay |
| Metacognition | Strategy debate + post-hoc `critique()` |
| Alignment/evals | Constitutional check + red-team battery; harder novel bench (15 cases) |

```bash
python -m cognitive_ai agi-gap --eval
python -m cognitive_ai agi-gap --ask "multi-step research photosynthesis then verify"
pytest tests/test_agi_gap_v116.py tests/test_agi_gap_v115.py -q
```

### What v1.15 adds — Proto-AGI gap upgrades (not AGI)

| Piece | Role |
|------|------|
| `agi_gap/world_knowledge.py` | Much larger combinatorial world/commonsense + open-world frames |
| `agi_gap/continual_skills.py` | Continual learning + auto-growing verified skill library |
| `agi_gap/deep_transfer.py` | Cross-domain schema transfer + analogy |
| `agi_gap/long_horizon.py` | Bounded long-horizon act→verify→reflect loops |
| `agi_gap/multimodal_hub.py` | Always-on speech/OCR/vision channel orchestration |
| `agi_gap/lifelong_memory.py` | Episodic consolidation into semantic/procedural memory |
| `agi_gap/metacognition.py` | Strategy selection, abstain, self-model limits |
| `agi_gap/alignment.py` | Product alignment guard + honesty badge |
| `benchmarks/hard_novel.py` | Hard evals on novel tasks **outside Finderbay** |

```bash
python -m cognitive_ai agi-gap --demo
python -m cognitive_ai agi-gap --eval
python -m cognitive_ai agi-gap --ask "multi-step research photosynthesis then verify"
python -m cognitive_ai bench
pytest tests/test_agi_gap_v115.py -q
```

> These modules **narrow the gap** toward broader intelligence. They do **not** achieve AGI.

### What v1.14 adds — ChatGPT-level non-LLM chat

| Piece | Role |
|------|------|
| `response/chatgpt_style.py` | Natural prose: overview, bullets, steps, yes/no, follow-ups |
| `nlu/smart_questions.py` | Dialogue acts, style (eli5/detailed), creative + math intents |
| `search/web_intel.py` | Query expansion, richer local KB, ChatGPT-style cited answers |
| `OpenDomainChat` | Poem/email/outline/brainstorm/math + smarter search escalation |
| Tests | Broad ChatGPT-style question battery (`test_chatgpt_level.py`) |

```bash
python -m cognitive_ai assistant --ask "Explain gravity in simple terms"
python -m cognitive_ai assistant --ask "Write a poem about Mars"
python -m cognitive_ai assistant --ask "Calculate 15% of 240"
python -m cognitive_ai assistant --ask "Search the web for photosynthesis"
pytest tests/test_chatgpt_level.py tests/test_smart_questions_and_search.py -q
```

> Still `human_level_agi=false`. For unrestricted cloud ChatGPT models, set `OPENAI_API_KEY` (auto-enables the LLM bridge). Offline, the non-LLM brain now answers in a ChatGPT-like helpful style.

### What v1.13 adds — Smarter questions + intelligent internet search

| Piece | Role |
|------|------|
| `nlu/smart_questions.py` | Paraphrase normalize, domain tag, answer policy |
| `nlu/questions.py` | Stronger WH/ask-form (search, inline WH, Hinglish) |
| `search/web_intel.py` | Rewrite → local KB → DuckDuckGo/Wikipedia → cite |
| OpenDomainChat | Uses smart frames + web escalation on KB miss |
| JARVIS `web.search` | Typed research tool with sources |

```bash
python -m cognitive_ai assistant --ask "What is gravity?"
python -m cognitive_ai assistant --ask "Search the web for photosynthesis"
pytest tests/test_smart_questions_and_search.py -q
```

### What v1.12 adds — Full-code strength upgrade

| Piece | Role |
|------|------|
| `intelligence/router.py` | Multi-signal SmartRouter (jarvis/vision/thermal/finderbay/coding/…) |
| `intelligence/confidence.py` | Honest confidence bands on answers |
| `intelligence/hardening.py` | Input harden + SelfCheck battery |
| JARVIS interpret | Hybrid regex + TF-IDF classifier |
| BM25 retrieval | Fuzzy + character n-gram boost (Hinglish/typos) |
| Verifier | More vision/report/payment checks |
| Assistant | Router-first dispatch + `self check` |

```bash
python -m cognitive_ai assistant --ask "self check"
python -m cognitive_ai diagnostics
```

### What v1.11 adds — Smarter vision + thermal & other modalities

| Piece | Role |
|------|------|
| `vision/modalities.py` | Detect RGB / **thermal** / IR / night / X-ray / depth / document / … |
| `vision/thermal.py` | Hotspots, cold spots, relative heat map, alerts |
| `vision/smart.py` | Scene type, risks, recommendations, follow-ups |
| Pipeline | modality → thermal (if IR) → see → analyze → understand → search → smart |

```bash
python -m cognitive_ai vision --task understand_scene --image ./warehouse_thermal_flir.jpg
python -m cognitive_ai vision --task thermal_analyze --image ./panel_thermal.jpg
python -m cognitive_ai vision --task detect_modality --image ./gate_night_vision.jpg
```

### What v1.10 adds — Vision understand loop

See→analyze→understand + object search; NLU `understand_scene`; vision tools.

### What v1.9 adds — complete JARVIS full code

| Piece | Role |
|------|------|
| `JarvisSystem` | Full facade: NLU + SQLite/FTS5 + policy + docs + dialogue + learning + voice |
| `nlu.py` | TF-IDF intent classifier + Hinglish slot extraction |
| `sqlite_memory.py` | SQLite facts + FTS5 BM25 document search |
| `learning.py` | Corrections → offline train → held-out eval → activate / rollback |
| `policy.py` | Trusted grants; planner cannot self-approve |
| `documents.py` | Create/verify report artifacts; decimal money |
| `dialogue.py` | Follow-ups (“other site”, “change the date”) |
| `adapters.py` | Files, Finderbay inventory, ASR/OCR bridges |
| Docs | `docs/JARVIS.md` · `FILE_INDEX.md` (full source listing) |

### What v1.8 added — JARVIS Tools & System Blueprint

Tool contracts, job store, provenance memory, material shortage skill, voice PTT, eval suite, CLI/API.

### Release zip

```bash
python scripts/package_release.py
# → dist/Complete_Non_LLM_AI_v1.18.0.zip
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
