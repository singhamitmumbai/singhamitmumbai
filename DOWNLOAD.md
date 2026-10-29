# Download Complete Non-LLM AI v1.27.0

Full packaged source (`human_level_agi=false`).

## Packaged zip (recommended)

| Item | Value |
|------|-------|
| File | [`releases/Complete_Non_LLM_AI_v1.27.0.zip`](releases/Complete_Non_LLM_AI_v1.27.0.zip) |
| Layout | `Complete_Non_LLM_AI/` + `RELEASE.txt` |
| Contents | `cognitive_ai`, tests, docs, scripts, Chat UI, CI — **465 files** |

### Direct GitHub links

- **Release zip (raw download):**  
  https://github.com/singhamitmumbai/singhamitmumbai/raw/cursor/complete-non-llm-ai-v04/releases/Complete_Non_LLM_AI_v1.27.0.zip

- **Branch source archive (GitHub):**  
  https://github.com/singhamitmumbai/singhamitmumbai/archive/refs/heads/cursor/complete-non-llm-ai-v04.zip

- **Pull request:**  
  https://github.com/singhamitmumbai/singhamitmumbai/pull/3

## Install after download

```bash
unzip Complete_Non_LLM_AI_v1.27.0.zip
cd Complete_Non_LLM_AI
pip install -e ".[dev]"
pytest -q
python -m cognitive_ai --version
python -m cognitive_ai chat --domain complete --port 8765
```

Optional:

```bash
export OPENAI_API_KEY=sk-...                 # ChatGPT-class chat
export COMPLETE_AI_COMPUTER_CONTROL=1        # gated computer control
export COMPLETE_AI_CONTINUOUS_LEARNING=1     # 24/7 learning
```

Rebuild the zip yourself anytime:

```bash
python scripts/package_release.py
# → dist/Complete_Non_LLM_AI_vVERSION.zip
```
