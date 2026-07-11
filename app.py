import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading DeepLabV3 on {device}...")
model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
model.eval()
print("Model loaded.")

def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

def generate_report(pixel_count):
    token = os.getenv("HF_TOKEN")
    if not token:
        return "Set HF_TOKEN in .env to enable AI reports."
    prompt = f"<|system|>You are an urban planning assistant.<|user|>AI detected {pixel_count} building pixels in a satellite image. Write a 2-sentence environmental warning memo to the county governor about urban encroachment on farmland. Suggest one policy action.<|assistant|>"
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 150}}
        )
        return r.json()[0]['generated_text'].split("<|assistant|>")[-1].strip()
    except Exception as e:
        return f"LLM error: {e}"

def analyze(image):
    img = image.convert("RGB").resize((512, 512))
    input_tensor = preprocess(img)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    pred = torch.argmax(output, dim=0).cpu().numpy()
    pixels = int((pred == 8).sum())
    mask = Image.fromarray(((pred == 8).astype(np.uint8) * 255)).resize(image.size)
    report = generate_report(pixels)
    return mask, report

with gr.Blocks(title="Urban Sprawl Detector") as demo:
    gr.Markdown("## Urban Sprawl Detector & AI Advisor")
    gr.Markdown("Upload a satellite image to detect buildings and get an AI-generated policy report.")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Upload Satellite Image", type="pil")
        with gr.Column():
            img_out = gr.Image(label="Building Detection Mask")
            text_out = gr.Textbox(label="AI Policy Advisory", lines=5)
    btn = gr.Button("Analyze", variant="primary")
    btn.click(fn=analyze, inputs=img_in, outputs=[img_out, text_out])
    gr.Markdown("Built for GeoML Datathon 2026")

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
