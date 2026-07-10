# GeoML Urban Sprawl Analyzer

Deployed on **Vercel** + **Hugging Face Spaces** + **Vercel Postgres**.

## Architecture

```
User Browser → Vercel (FastAPI + HTML frontend)
                  ↓
             Vercel Postgres (Neon) — stores analysis history
                  ↓
             Hugging Face Spaces — DeepLabV3 ML inference
                  ↓
             Hugging Face API — LLM report generation (Zephyr-7B)
```

## Project Structure

```
GeoML_WebApp/
├── api/index.py           # FastAPI backend (Vercel serverless)
├── api/database.py        # PostgreSQL connection (SQLAlchemy)
├── api/models.py          # Database models
├── api/schemas.py         # Pydantic schemas
├── utils/model_loader.py  # DeepLabV3 model
├── utils/llm_client.py    # Hugging Face Inference API client
├── hf_spaces/app.py       # Gradio ML backend (HF Spaces)
├── data/fetch_real_image.py  # Fetch real satellite tiles
├── vercel.json            # Vercel config
├── requirements.txt       # Python deps
└── .env.example           # Environment variable template
```

## Deployment Steps

### 1. Deploy ML Backend to Hugging Face Spaces

```bash
# Create a Space at huggingface.co/new-space
# SDK: Gradio, Hardware: CPU (free) or GPU
# Upload hf_spaces/app.py and hf_spaces/requirements.txt
```

### 2. Deploy Vercel App

```bash
# Push to GitHub, then:
vercel --prod
```

### 3. Set Environment Variables (Vercel Dashboard)

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `postgresql://...` (Vercel Postgres) |
| `HF_TOKEN` | `hf_...` (Hugging Face token) |
| `HF_SPACES_URL` | `https://your-space.hf.space` |

### 4. Fetch Real Satellite Image

```bash
cd data
python fetch_real_image.py
# Output: real_satellite_image.png
```

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload
```
