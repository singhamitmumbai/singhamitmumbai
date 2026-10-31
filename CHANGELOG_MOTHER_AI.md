# CHANGELOG — Mother AI

## 1.29.0 — Second-pass cognition upgrades

### Added
- Structured `MotherWorldModel` (entities/resources + cue prediction error)
- `RoleLibrary` proposer/critic/verifier rule IDs
- Budgeted `ContinuousCognition` (`cognition` CLI)
- Isolated `SelfModLab` with human review + rollback (`selfmod` CLI)
- Expanded FinderBench (`finderbench_v2`, soft holdout)

### Honesty
- `human_level_agi=false` — self-mod never auto-patches live source

## 1.28.0 — Finderbay Mother AI integration

### Fixed
- Natural language in default REPL no longer yields `Unknown command` for greetings / free-form chat.

### Added
- `cognitive_ai/mother/` integrated runtime (gateway, goals, beliefs, workspace, events, MotherBrain, FinderBench)
- CLI modes: `mother`, `diagnose`, `benchmark`, `status`
- Default `python -m cognitive_ai` launches Mother AI
- Audit reports under `reports/mother_ai/`
- Deliverables: `MOTHER_AI_UPGRADE_REPORT.md`, `MOTHER_AI_FINAL_RESULTS.json`, `MOTHER_AI_LIMITATIONS.md`, …

### Honesty
- `human_level_agi=false`
