# How to upgrade Complete AI

Complete AI is versioned (`cognitive_ai/version.py`). Upgrading means one of:

1. **Software upgrade** — pull a newer release of the code  
2. **Capability upgrade** — turn on optional engines (speech, OCR, vision, LLM)  
3. **Knowledge / skill upgrade** — teach aliases, corrections, procedures inside the product  
4. **Product upgrade** — ship a new minor (think / KB / tools) via the full-test gate  

Still `human_level_agi=false` after any upgrade.

---

## 1) Upgrade the code (new zip / git)

```bash
# From a release zip
unzip Complete_Non_LLM_AI_vX.Y.Z.zip
cd Complete_Non_LLM_AI
pip install -e ".[dev]"

# Or from git on this branch
git pull
pip install -e ".[dev]"
```

Verify:

```bash
python -m cognitive_ai --version
python -m cognitive_ai diagnostics
python -m cognitive_ai completeness
python -m cognitive_ai full-test
pytest -q
```

Rebuild a distributable zip after you upgrade sources:

```bash
python scripts/package_release.py
# → dist/Complete_Non_LLM_AI_vVERSION.zip
```

Session memory (Complete AI) lives under `data/complete_ai/`. Back it up before a major bump if you care about chat memory.

---

## 2) Upgrade capabilities (optional engines)

| Wish | What to install / set | Check |
|------|------------------------|--------|
| Better general chat | `export OPENAI_API_KEY=sk-...` | `llm_at_runtime=true` in status |
| Offline speech (EN/HI) | `pip install -e ".[perception]"` + Vosk models via `model_registry` | `list_models()` / Hindi factory |
| OCR + tables | Tesseract system pkgs + `.[perception]` | Layout OCR on invoices |
| Real YOLO vision | `pip install -e ".[vision]"` | `vision status` shows ultralytics |
| Computer control (detailed) | `export COMPLETE_AI_COMPUTER_CONTROL=1` (+ optional `COMPLETE_AI_COMPUTER_SHELL=1`) | `computer help` · see `COMPUTER_CONTROL.md` |
| Everything | `pip install -e ".[all]"` | diagnostics + full-test |

Speech models (no LLM):

```bash
python - <<'PY'
from cognitive_ai.perception.model_registry import list_models, ensure_model
print(list_models())
# ensure_model("hi-small", download=True)   # downloads Hindi pack when needed
PY
```

Disable cloud chat anytime:

```bash
export COMPLETE_AI_DISABLE_LLM=1
```

---

## 3) Upgrade knowledge inside a running system

The product can improve **without** claiming AGI:

| Wish | How |
|------|-----|
| Remember a fact | Assistant: `remember that my warehouse is Central Store` |
| Alias / synonym | `learn alias unpaid outstanding` or “X is also called Y” |
| Correction | `correct: <wrong> → <right>` (review before release) |
| Plan before act | `plan Show unpaid invoices for Project A` |
| Proto skills / transfer | `python -m cognitive_ai agi-gap --ask "…"` |
| Finderbay auth roles | `login acct1 password acct-pass` then payments / POs |

Internal loop (bounded): **failure → review → candidate → evaluate → release**  
(`cognitive_ai/learning_sys/improvement.py`). Unverified self-labels are rejected.

---

## 4) Wish-list upgrades (next meaningful product steps)

If you want to **upgrade the system further**, prioritize in this order:

1. **Richer Finderbay workflows** — payments/contracts/PO edge cases, multi-user UI, audit logs  
2. **Perception packs** — ship or script Hindi/EN model download; mic streaming in Chat UI  
3. **OCR production path** — PDF multi-page + table export into invoice fields  
4. **Stronger composition** — raise SCAN holdout accuracy (NLU slots for filters/exclusions)  
5. **Lifelong memory defaults** — persist proto-AGI lifelong store across restarts out of the box  
6. **Eval gates in CI** — run `full-test` on every PR  

Do **not** treat optional LLM chat as “upgrading to AGI”. With a key you get ChatGPT-class *chat*; local engines stay non-LLM; honesty flag stays false.

---

## Quick decision guide

| You wish to… | Do this |
|--------------|---------|
| Get the latest code | New zip / `git pull` + `pip install -e ".[dev]"` + `full-test` |
| Talk like ChatGPT | Set `OPENAI_API_KEY` |
| Hear / read documents better | `.[perception]` + models + Tesseract |
| See objects in images | `.[vision]` |
| Teach the business brain | remember / learn alias / correct / plan |
| Prove the upgrade worked | `diagnostics` · `completeness` · `full-test` · `pytest -q` |

```bash
python -m cognitive_ai full-test
# ok=true, human_level_agi=false
```
