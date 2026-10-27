# Install — Complete Non-LLM AI v1.2.1

## Requirements

- Python 3.10+ (3.12 recommended)
- No pretrained general-purpose LLM is required or used at runtime

## Quick install (core)

```bash
unzip Complete_Non_LLM_AI_v1.2.1.zip
cd Complete_Non_LLM_AI
pip install -e ".[dev]"
pytest -q
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
# → dist/Complete_Non_LLM_AI_v1.2.1.zip
```

## Honest limits

- Not human-level AGI (`human_level_agi: false`)
- Computer vision uses CNN / classical CV — not ChatGPT-style generation
- Stub vision mode works without Ultralytics; install `[vision]` for real detections
