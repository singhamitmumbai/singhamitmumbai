# Symbolic Cognitive AI

A runnable **non-LLM** “brain”: structured knowledge, beliefs, rule-based inference, means-ends planning, a self-model, an autonomous think loop, and simple symbolic learning.

No OpenAI/Anthropic APIs, no local chat models, no transformers generation — cognition is **symbolic / algorithmic**.

## What “thinking” means here

Each think cycle is discrete and inspectable:

1. **Perceive** — ingest structured triples into working memory / KB  
2. **Update beliefs** — confidence-weighted situation model  
3. **Reason** — forward chaining (and on-demand backward proofs) over production rules  
4. **Plan** — goals → subgoals → actions (means-ends / hierarchical)  
5. **Act / speak** — apply action effects; emit explanations  
6. **Reflect / learn** — record thoughts; write facts/rules or failure lessons  

This is classical cognitive-architecture style reasoning, not next-token prediction.

## Quick start

```bash
# optional: install package editable + pytest
pip install -e ".[dev]"

# offline demo (zero network)
python -m cognitive_ai --demo
# or
python scripts/demo_brain.py

# interactive REPL
python -m cognitive_ai --domain animals

# non-interactive think
python -m cognitive_ai --domain kitchen --goal "Make omelette|omelette status ready" --think 6
```

### REPL examples

```
brain> domain animals
brain> reason
brain> prove tweety can fly
brain> why tweety can fly
brain> goal Feed tweety | tweety fed true
brain> think 4
brain> ask What do you know?
brain> ask What are you trying to do?
brain> dump
brain> save /tmp/mind.json
```

## Architecture

| Module | Role |
|--------|------|
| `cognitive_ai/knowledge.py` | Triple store + rules; JSON/SQLite persistence |
| `cognitive_ai/beliefs.py` | Working memory, open questions, confidence |
| `cognitive_ai/inference.py` | Forward & backward chaining + `why` |
| `cognitive_ai/planner.py` | Goals, actions, means-ends planner |
| `cognitive_ai/self_model.py` | Introspection (“what do you know?”, limits, thoughts) |
| `cognitive_ai/think_loop.py` | Autonomous perceive→…→reflect cycle |
| `cognitive_ai/learning.py` | Fact/rule acquisition & failed-plan lessons |
| `cognitive_ai/brain.py` | Wires modules together |
| `cognitive_ai/domains.py` | Demo domains: `animals`, `blocks`, `kitchen` |
| `cognitive_ai/cli.py` | CLI + REPL |

## Tests

```bash
pytest -q
```

Covers inference, planning, self-model, think loop, and persistence.

## Dependencies

Runtime default: **Python 3.10+ stdlib only** (`sqlite3`, `json`, …).  
Dev optional: `pytest`. No ML/LLM packages required.
