# Install — Complete AI v1.6.0

## Requirements

- Python 3.10+ (3.12 recommended)
- For ChatGPT-class general chat: `OPENAI_API_KEY` (OpenAI-compatible)
- Specialist engines (Finderbay / vision / coding) run locally without an LLM

## Quick install

```bash
unzip Complete_Non_LLM_AI_v1.6.0.zip
cd Complete_Non_LLM_AI
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...   # enables Complete AI general chat + tools
pytest -q
python -m cognitive_ai complete-ai --demo
python -m cognitive_ai diagnostics
python -m cognitive_ai completeness
```

## Optional extras

```bash
# Speech (Vosk) + OCR (Tesseract wrappers)
pip install -e ".[perception]"
# Also install system packages: tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin
# Optional: download Vosk models into models/ (see cognitive_ai/perception/model_registry.py)

# Real YOLO CNN inference (Ultralytics + OpenCV)
pip install -e ".[vision]"
# First run may download yolo11n.pt

# Everything
pip install -e ".[all]"
```

## Run

```bash
python -m cognitive_ai finderbay --demo
python -m cognitive_ai general --demo
python -m cognitive_ai vision --task vision_status
python -m cognitive_ai finderbay --ask "Count people in image ./crowd_people.jpg"
python -m cognitive_ai agent --demo
python -m cognitive_ai code --ask "generate function add(a, b)"
python -m cognitive_ai blueprint
python scripts/demo_vision.py
```

## Zip rebuild

```bash
python scripts/package_release.py
# → dist/Complete_Non_LLM_AI_v1.6.0.zip
```

## ChatGPT-class general chat (Complete AI default open chat)

```bash
export OPENAI_API_KEY=sk-...
# optional: COMPLETE_AI_LLM_MODEL=gpt-4o-mini
python -m cognitive_ai complete-ai --ask "Explain recursion simply"
python -m cognitive_ai complete-ai --mode llm --ask "Write a cover letter"
# force symbolic-only: COMPLETE_AI_DISABLE_LLM=1
```

With a key, general chat uses your ChatGPT-class model (`llm_at_runtime=true`).  
Business / vision / coding still use local engines.

## Limits (status flags)

- `human_level_agi: false` — Complete AI does not claim AGI
- Vision: CNN / classical CV (`pip install -e ".[vision]"` for real YOLO)
