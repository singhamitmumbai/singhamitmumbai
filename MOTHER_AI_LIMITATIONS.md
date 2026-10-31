# Finderbay Mother AI — Limitations (v1.29.0)

**human_level_agi=false**

## Do not claim

- AGI / human-level intelligence / consciousness / self-awareness
- Fully autonomous self-improvement or live code self-modification
- Human-equivalent conversation
- Production-grade perception on real cameras/mics without measured quality

## Known technical limits

1. **Proposer / Critic / Verifier** are explicit rule libraries inside one process (`roles.py`) — not independently trained adversarial systems.
2. **World-model** (`world_model.py`) now tracks structured entities/resources + cue-based prediction error; it is still a lightweight scratch model, not learned dynamics.
3. **Self-modification lab** isolates candidates with evaluate/review/promote/rollback — promotion never patches live product source automatically; human reviewer required.
4. **FinderBench** expanded (~30+ cases + soft holdout) — still not a full 100-item / hidden-holdout AGI suite.
5. **Overlapping runtimes** (Brain, CompleteAI, Assistant, JARVIS, ProtoAGI) still exist; MotherBrain coordinates them rather than deleting them.
6. **Hinglish** support is incremental pattern coverage — not full multilingual NLU.
7. **Continuous cognition** is budget-capped (`max_cycles` / `max_seconds` / tool / attention). Not unbounded autonomous AGI.

## Reliability boundary

Irreversible computer-control / shell effects remain gated. Thinking and planning may run freely; external side effects require capability gates and verification. Self-mod lab forbids `model`/`self`/`auto` reviewers.
