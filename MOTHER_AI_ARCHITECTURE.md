# Finderbay Mother AI — Architecture (v1.28.0)

```
User input
   ↓
InputGateway (normalize, dialogue-act, intent, entities)
   ↓
MotherBrain cognitive cycle
   ├─ GlobalWorkspace (bounded)
   ├─ GoalManager (persistent)
   ├─ BeliefStore (persistent)
   ├─ EventBus (typed)
   ├─ Proposer → Critic → Verifier/Select
   └─ Execute via backends:
        • symbolic Brain / dialogue
        • CompleteAI (Finderbay/tools/coding/vision/chat)
        • inspect/diagnostics
   ↓
Observe → prediction error → learn/beliefs/goals
   ↓
Response + mother_cycle trace
```

## Design rules

- Preserve existing working modules; integrate rather than rewrite.
- Every cycle is observable (`mother_cycle.stages`).
- Never claim AGI; `human_level_agi=false` on all public payloads.
