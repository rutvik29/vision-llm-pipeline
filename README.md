# 👁️ Vision LLM Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-0075DB?style=flat)](https://ultralytics.com)
[![GPT-4V](https://img.shields.io/badge/GPT--4V-Vision-412991?style=flat&logo=openai)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end vision + LLM pipeline** — YOLOv8 detects objects, GPT-4V understands scenes, and a FastAPI service answers visual questions in real time.

## ✨ Highlights

- 🎯 **YOLOv8 detection** — COCO-pretrained with custom fine-tuning support
- 👁️ **GPT-4V scene understanding** — describe, caption, answer questions about images
- ❓ **Visual QA** — ask natural language questions about any image
- 🎬 **Video processing** — frame-by-frame analysis with temporal context
- ⚡ **Streaming API** — FastAPI with Server-Sent Events for progressive results
- 📊 **Confidence filtering** — configurable detection thresholds per class

## Quick Start

```bash
git clone https://github.com/rutvik29/vision-llm-pipeline
cd vision-llm-pipeline
pip install -r requirements.txt
cp .env.example .env

# Analyze an image
python analyze.py --image ./examples/street.jpg --question "How many cars are there?"

# Start API
python -m src.api.server  # :8006
```

## License
MIT © Rutvik Trivedi
