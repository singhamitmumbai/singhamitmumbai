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
| 24/7 self-learning | `export COMPLETE_AI_CONTINUOUS_LEARNING=1` | `learn --ticks 3` · see `SELF_LEARNING.md` |
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
| Learn 24/7 automatically | `export COMPLETE_AI_CONTINUOUS_LEARNING=1` then `python -m cognitive_ai learn --daemon` (or Assistant: `start continuous learning`) |
| Finderbay auth roles | `login acct1 password acct-pass` then payments / POs |

Internal loop (bounded): **failure → review → candidate → evaluate → release**  
(`cognitive_ai/learning_sys/improvement.py`). Unverified self-labels are rejected.

---

## 4) What v1.26 already upgraded

Shipped from the prior wish-list:

1. **Finderbay audit logs** — `domains/finderbay/audit.py` · `Show audit log`  
2. **Lifelong memory defaults** — auto persist/load (`COMPLETE_AI_LIFELONG_PATH`, on by default)  
3. **Stronger composition** — between-ranges, under≠overdue, stock qty, exclusions → SCAN ch9–ch11  
4. **OCR→invoice export** — `LayoutOCR.export_invoice_fields`  
5. **CI eval gate** — `.github/workflows/ci.yml`  
6. **Perception bootstrap** — `scripts/ensure_perception_models.py`  

## 5) Wish-list upgrades (next meaningful product steps)

1. **Richer Finderbay UI** — multi-user Chat UI session tokens + audit viewer  
2. **Perception packs** — mic streaming wired into Chat UI; model download UX  
3. **OCR PDF path** — native multi-page PDF → invoice import into memory  
4. **SCAN holdout ≥0.85** — more compositional operators (OR filters, nested exclusions)  
5. **Continuous learning promotions** — human review queue for mined candidates  

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
| Search past chat turns | `search messages for invoice` · Chat UI sidebar search |
| Rewrite / vary a message | `vary message: …` · **Vary** button in Chat UI |
| Prove the upgrade worked | `diagnostics` · `completeness` · `full-test` · `pytest -q` |

```bash
python -m cognitive_ai full-test
# ok=true, human_level_agi=false
```
