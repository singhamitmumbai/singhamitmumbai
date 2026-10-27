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
| `cognitive_ai/domains/finderbay` | Invoices, deliveries, materials, tasks, payments, contracts, auth |
| `cognitive_ai/commonsense` | Curated multi-thousand fact KB + reasoner + eval suite |
| `cognitive_ai/benchmarks` | SCAN-style composition + skill-transfer holdouts |
| `cognitive_ai/general` | `GeneralNonLLMAI` router (Finderbay ↔ open-domain) |
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

## v0.8 operational expansion

1. `runtime/alerts.py` proactive monitor (overdue / low stock / expiry / pending)  
2. Finderbay vendors + `purchase_orders` service (auth-gated writes)  
3. Multi-turn `ClarificationManager` slot-filling in dialogue tracker  
4. Causal common-sense `why()` + Hindi-ish open-domain patterns  
5. Stdlib REST API (`cognitive_ai api`) + CSV/JSON `ReportExporter`

## v0.7 capability expansion

1. Large common-sense KB generator (`commonsense/generate_kb.py`) + eval suite  
2. Speech model registry (en/hi/en-in) + `StreamingASR` / optional mic loop  
3. `LayoutOCR` multi-page invoices + table heuristics  
4. Finderbay `auth`, `payments`, `contracts` services  
5. `benchmarks/scan_like.py` + `transfer_bench.py` (`cognitive_ai bench`)

## v0.6 generality upgrades

- **Open-domain chat** via curated common-sense KB + pattern router (`cognitive_ai/commonsense`)
- **Production speech**: Vosk offline ASR (`perception/speech_production.py`)
- **Production OCR**: Tesseract 5 eng+hin (`perception/ocr_production.py`)
- **`GeneralNonLLMAI`**: routes Finderbay tasks ↔ open-domain chat
- **Skill transfer** helper across projects
- Honest status flags: `human_level_agi=false`, `open_domain_chat=true`

## v0.5 completion

- Versioned **dataset registry** with source-group split locks (no leakage)
- **Calibration** helpers (Brier + reliability bins) and uncertainty policies
- **Recovery manager** for durable checkpoints / OUTCOME_UNKNOWN reconcile
- CSV/JSON **data adapters** for Finderbay registers
- HTN method library for workflow decompositions
- Hindi/Hinglish invoice & pending realization
- Alias learning (`Warehouse B is now called Central Store`)
- Config object (`SystemConfig`) wired through the orchestrator

## v0.4 upgrades

- BM25 retrieval with synonym expansion; claim dependency invalidation
- Entity linker (projects, materials, aliases, invoices)
- Tool registry (typed args, permissions, side-effects)
- Operators: `reserve_stock`, `create_task`, `list_tasks`, `ocr_extract`
- Runtime capability model + latency/coverage monitor
- Finderbay chat UI mode (`--domain finderbay`)
- Expanded eval suite (invoices, deliveries, materials, reserve, tasks, OCR)

## Quick start

```bash
pip install -e ".[dev]"
python -m cognitive_ai finderbay --demo
python -m cognitive_ai chat --domain finderbay
python -m cognitive_ai eval
pytest -q
```
