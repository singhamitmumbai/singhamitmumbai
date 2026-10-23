# Symbolic Cognitive AI + Cognitive OS

A runnable **non-LLM** “brain” and a self-contained **Cognitive OS** around it: structured knowledge, beliefs, rule-based inference, means-ends planning, a self-model, an autonomous think loop, symbolic learning, and an interactive OS shell.

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

## Launch the chat UI

Browser chat against the symbolic brain (stdlib HTTP server, no LLM):

```bash
python -m cognitive_ai chat
# or
python -m cognitive_ai.ui
python scripts/chat_ui.py

# then open http://127.0.0.1:8765/
```

## Launch the Cognitive OS

One-shot boot into an interactive OS shell (brain + filesystem store + console + optional background cognition):

```bash
pip install -e ".[dev]"

# Interactive OS
python -m cognitive_ai os
# or
python scripts/boot_os.py

# Scripted OS demo (non-interactive proof)
python -m cognitive_ai os --os-demo
python -m cognitive_ai.os --demo
python scripts/boot_os.py --demo
```

OS store defaults to `~/.cognitive_os/` (`knowledge.json`, `console.log`, `os_state.json`). Override with `--root /path`.

### OS shell commands

| Command | Purpose |
|---------|---------|
| `boot [domain]` | Start brain + services |
| `status` | System / brain / scheduler status |
| `think [N]` | Run N think cycles |
| `ask …` | Talk to the brain |
| `assert s p o` | Assert a fact |
| `goal desc \| s p o` | Set a goal |
| `ps` | Active goals (process table) |
| `mem` | Knowledge + beliefs |
| `why s p o` | Explain a conclusion |
| `sched start\|stop\|tick` | Background cognition |
| `dmesg` | Console log |
| `save` / `shutdown` | Persist and halt |

Prompt looks like: `cos>`

## Brain-only quick start

```bash
python -m cognitive_ai --demo
python -m cognitive_ai --domain animals          # brain REPL (brain>)
python -m cognitive_ai --domain kitchen --goal "Make omelette|omelette status ready" --think 6
pytest -q
```

## Architecture

| Module | Role |
|--------|------|
| `cognitive_ai/knowledge.py` | Triple store + rules; JSON/SQLite persistence |
| `cognitive_ai/beliefs.py` | Working memory, open questions, confidence |
| `cognitive_ai/inference.py` | Forward & backward chaining + `why` |
| `cognitive_ai/planner.py` | Goals, actions, means-ends planner |
| `cognitive_ai/self_model.py` | Introspection |
| `cognitive_ai/think_loop.py` | Autonomous perceive→…→reflect cycle |
| `cognitive_ai/learning.py` | Fact/rule acquisition & failed-plan lessons |
| `cognitive_ai/brain.py` | Wires modules together |
| `cognitive_ai/domains.py` | Demo domains: `animals`, `blocks`, `kitchen` |
| `cognitive_ai/cli.py` | CLI + brain REPL + `os` entry |
| `cognitive_ai/os/` | **Cognitive OS** — boot, services, shell |

### OS services

- **Store** — filesystem-backed knowledge + mind dumps  
- **Scheduler** — think loop as background “process”  
- **Console** — message/log ring buffer (`dmesg`)  

## Dependencies

Runtime default: **Python 3.10+ stdlib only**.  
Dev optional: `pytest`. No ML/LLM packages required.
