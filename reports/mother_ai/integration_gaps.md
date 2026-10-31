# Integration gaps (post Mother v1.29)

## Fixed (v1.28–v1.29)
- Bare REPL `Unknown command` for `hii` / `how are you` → Mother dialogue
- Unified cognitive cycle with observable stages
- Persistent goals/beliefs under `data/complete_ai/mother/`
- Structured world-model with cue-based prediction error + resource burn
- Explicit RoleLibrary critic/verifier rules
- Budgeted continuous cognition ticks
- Self-mod lab with isolated candidates, human review, rollback
- FinderBench expanded (`finderbench_v2`) with soft holdout

## Remaining
- Proposer/critic/verifier are rule libraries, not independently trained models
- World-model is lightweight scratch + domain transitions — not learned dynamics
- Self-mod does not auto-patch live product source (by design)
- FinderBench is still not a 100-item hidden AGI suite
- Perception quality not re-measured on real cameras
- Overlapping runtimes (Brain / CompleteAI / Assistant / JARVIS) still coexist
