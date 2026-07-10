import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import base64
import io
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(image):
    img = image.convert("RGB")
    orig_size = img.size
    input_tensor = preprocess(img)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    pred = torch.argmax(output, dim=0).cpu().numpy()

    building_mask = (pred == 8).astype(np.uint8)
    building_pixels = int(building_mask.sum())
    total_pixels = building_mask.size
    urban_pct = int((building_pixels / total_pixels) * 100)

    mask_pil = Image.fromarray(building_mask * 255).resize(orig_size)

    # Convert mask to base64 for API response
    buffered = io.BytesIO()
    mask_pil.save(buffered, format="PNG")
    mask_b64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        "building_pixels": building_pixels,
        "total_pixels": total_pixels,
        "urban_percentage": urban_pct,
        "mask_b64": mask_b64,
    }

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🏙️ GeoML Building Detector (ML Backend)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Satellite Image", type="pil")
        with gr.Column():
            output_img = gr.Image(label="Building Mask")
            stats = gr.JSON(label="Statistics")
    btn = gr.Button("🔍 Analyze")
    btn.click(
        fn=lambda img: (predict(img)["mask_b64"], predict(img)),
        inputs=input_img,
        outputs=[output_img, stats]
    )

# Expose API endpoint for Vercel
gr.mount_gradio_app(demo, "/api/predict")
