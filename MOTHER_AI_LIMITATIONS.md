# Finderbay Mother AI — Limitations (v1.28.0)

**human_level_agi=false**

## Do not claim

- AGI / human-level intelligence / consciousness / self-awareness
- Fully autonomous self-improvement
- Human-equivalent conversation
- Production-grade perception on real cameras/mics without measured quality

## Known technical limits

1. **Proposer / Critic / Verifier** are rule/heuristic roles inside one process — not independently trained adversarial systems.
2. **World-model prediction error** uses lexical overlap between predicted and observed text — not structured dynamics.
3. **Concept discovery, causal learning, self-modification lab** from the master blueprint are only partially scaffolded (events/learning_log exist; full pipelines are not complete).
4. **FinderBench** currently ships as a smoke suite (~15 cases), not the full 100-item / hidden-holdout suite.
5. **Overlapping runtimes** (Brain, CompleteAI, Assistant, JARVIS, ProtoAGI) still exist; MotherBrain coordinates them rather than deleting them.
6. **Hinglish** support is incremental pattern coverage — not full multilingual NLU.
7. **Continuous cognition** budgeted autonomous cycles are not the default interactive path (use existing `learn --daemon` / OS scheduler separately).

## Reliability boundary

Irreversible computer-control / shell effects remain gated. Thinking and planning may run freely; external side effects require capability gates and verification.
