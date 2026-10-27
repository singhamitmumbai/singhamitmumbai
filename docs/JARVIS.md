# JARVIS Tools & System Blueprint — Finderbay implementation

Non-LLM controller with optional LLM. `human_level_agi=false`.

## Control loop

Capture → Interpret → Resolve → Plan → Execute → Verify → Respond → Record

## Package layout (`cognitive_ai/jarvis/`)

| File | Blueprint | Role |
|------|-----------|------|
| `system.py` | §§1–14 | Full facade |
| `pipeline.py` | §3 | Request pipeline |
| `contracts.py` | §8, §11 | Tool contracts, job states, facts |
| `registry.py` | §6 | Callable tools |
| `jobs.py` | §11 | Durable job store |
| `skills.py` | §5D, §13 | Material shortage workflow |
| `provenance.py` | §7 | Five memory kinds |
| `sqlite_memory.py` | §7 | SQLite + FTS5 |
| `nlu.py` | §5 | TF-IDF intent + slots |
| `dialogue.py` | §5C | Follow-ups / referents |
| `policy.py` | §8 | Trusted authorization |
| `documents.py` | §10 | Extract / create / verify reports |
| `adapters.py` | §9–§10 | Finderbay + perception bridges |
| `voice.py` | §4 | PTT / VAD / ASR / TTS adapters |
| `learning.py` | §12 | Corrections → train → eval → activate |
| `eval_suite.py` | §14 | Evaluation criteria |

## CLI

```bash
python -m cognitive_ai jarvis --eval
python -m cognitive_ai jarvis --ask "Prepare tomorrow's material shortage report for Site-7"
python -m cognitive_ai jarvis --demo
```

## API

`POST /jarvis` with `{"text": "..."}` or `{"action":"eval"}`.
