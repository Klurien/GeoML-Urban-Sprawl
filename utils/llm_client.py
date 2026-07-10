import os
import json
import urllib.request
import urllib.error

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_URL = "https://router.huggingface.co/v1/chat/completions"

def generate_report(building_pixels: int, urban_pct: int) -> str:
    if not HF_TOKEN:
        return "Configuration error: HF_TOKEN not set."

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful urban planning assistant."
            },
            {
                "role": "user",
                "content": f"My AI analyzed a satellite image and detected {building_pixels} pixels of urban concrete, which represents {urban_pct}% of the image area. Write a short 2-paragraph official warning memo to the County Governor about this rapid urbanization and its potential impact on rural agriculture. Provide 1 suggested policy action."
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return str(result)
    except Exception as e:
        return f"Report generation unavailable: {str(e)[:100]}"
