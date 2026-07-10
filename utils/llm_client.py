import os
import httpx

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

async def generate_report(building_pixels: int, urban_pct: int) -> str:
    if not HF_TOKEN:
        return "⚠️ Configuration error: HF_TOKEN not set."

    prompt = f"""<|system|>
You are a helpful urban planning assistant.
<|user|>
My AI analyzed a satellite image and detected {building_pixels} pixels of urban concrete, which represents {urban_pct}% of the image area.
Write a short 2-paragraph official warning memo to the County Governor about this rapid urbanization and its potential impact on rural agriculture. Provide 1 suggested policy action.
<|assistant|>"""

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return str(result)
        except Exception as e:
            return f"⚠️ Report generation unavailable: {str(e)[:100]}"
