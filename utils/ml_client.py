import os
import base64
import httpx
from io import BytesIO
from PIL import Image
import numpy as np

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

def segment_buildings(image_bytes: bytes, max_retries: int = 2) -> dict:
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set"}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for attempt in range(max_retries):
        with httpx.Client(timeout=120.0) as client:
            response = client.post(API_URL, headers=headers, content=image_bytes)

        if response.status_code == 503 and "loading" in response.text.lower():
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue

        response.raise_for_status()
        segments = response.json()
        break

    if not isinstance(segments, list):
        return {"error": f"Unexpected HF response: {str(segments)[:200]}"}

    img = Image.open(BytesIO(image_bytes))
    w, h = img.size
    combined = np.zeros((h, w), dtype=np.uint8)

    for seg in segments:
        label = seg.get("label", "")
        if "building" in label.lower():
            mask_b64 = seg.get("mask")
            if not mask_b64:
                continue
            mask_data = base64.b64decode(mask_b64)
            mask = np.array(Image.open(BytesIO(mask_data)).convert("L"))
            binary = (mask > 128).astype(np.uint8)
            combined = np.maximum(combined, binary)

    building_pixels = int(np.sum(combined > 0))
    total_pixels = w * h
    urban_pct = round(building_pixels / total_pixels * 100, 1) if total_pixels > 0 else 0.0

    mask_pil = Image.fromarray((combined * 255).astype(np.uint8))
    buf = BytesIO()
    mask_pil.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "building_pixels": building_pixels,
        "total_pixels": total_pixels,
        "urban_percentage": urban_pct,
        "mask_b64": mask_b64,
    }
