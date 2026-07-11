---
title: GeoML Urban Sprawl Detector
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# GeoML Urban Sprawl Detector

Satellite image building detection with AI-generated policy reports.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python data/fetch_real_image.py
python app.py
```

## Free Public URL (no deploy needed)

Gradio has built-in `share=True` to create a temporary public URL for free (valid ~72h):

```bash
python -c "
import app
app.demo.launch(share=True)
"
```

## Architecture

```
Browser → Gradio → DeepLabV3 (on CPU/GPU) → HF Inference API (Zephyr-7B) → Mask + Report
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio web app with model + LLM |
| `data/fetch_real_image.py` | Download real satellite image from GeoJSON bounds |
| `data/geoml_hackerthon.geojson` | Labeled urban polygons |
| `.env` | `HF_TOKEN=hf_...` |
