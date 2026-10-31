# Finderbay Mother AI — Architecture (v1.29.0)

```
User input
   ↓
InputGateway (normalize, dialogue-act, intent, entities)
   ↓
MotherBrain cognitive cycle (mother_brain_v2)
   ├─ GlobalWorkspace (bounded)
   ├─ GoalManager (persistent)
   ├─ BeliefStore (persistent)
   ├─ EventBus (typed)
   ├─ MotherWorldModel (entities/resources + predict/observe)
   ├─ RoleLibrary: Proposer → Critic → Verifier/Select
   ├─ ContinuousCognition (optional budgeted ticks)
   ├─ SelfModLab (isolated candidates; human review)
   └─ Execute via backends:
        • symbolic Brain / dialogue
        • CompleteAI (Finderbay/tools/coding/vision/chat)
        • inspect/diagnostics
   ↓
Observe → structured + lexical prediction error → learn/beliefs/goals
   (+ optional isolated self-mod candidate on high error)
   ↓
Response + mother_cycle trace
```

## Design rules

- Preserve existing working modules; integrate rather than rewrite.
- Every cycle is observable (`mother_cycle.stages`).
- Never claim AGI; `human_level_agi=false` on all public payloads.
- Self-mod never auto-applies to live product source; `model`/`self`/`auto` reviewers forbidden.
