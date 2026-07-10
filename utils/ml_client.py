import base64
from io import BytesIO
from PIL import Image
import numpy as np

def segment_buildings(image_bytes: bytes) -> dict:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Urban/concrete heuristic in satellite imagery:
    # - Low saturation (grey-ish): max(r,g,b) - min(r,g,b) < threshold
    # - Medium brightness: mean(r,g,b) between 80 and 220
    # - Low green channel (not vegetation)
    saturation = np.max(arr, axis=2) - np.min(arr, axis=2)
    brightness = np.mean(arr, axis=2)

    urban = (
        (saturation < 45) &
        (brightness > 80) &
        (brightness < 220) &
        (g < 180)  # not strongly green
    ).astype(np.uint8)

    building_pixels = int(np.sum(urban > 0))
    total_pixels = w * h
    urban_pct = round(building_pixels / total_pixels * 100, 1) if total_pixels > 0 else 0.0

    mask_pil = Image.fromarray((urban * 255).astype(np.uint8), mode="L")
    buf = BytesIO()
    mask_pil.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "building_pixels": building_pixels,
        "total_pixels": total_pixels,
        "urban_percentage": urban_pct,
        "mask_b64": mask_b64,
    }
