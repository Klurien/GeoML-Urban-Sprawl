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

## Deploy on Hugging Face Spaces

1. Go to https://huggingface.co/new-space
2. Name: `geo-urban-detector`, SDK: **Gradio**
3. Upload `app.py` and `requirements.txt`
4. Settings → Repository secrets → add `HF_TOKEN` with your Hugging Face token
5. Wait 2 min for build

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
