import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import io
import base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"🚀 Loading DeepLabV3 on {device}...")
        _model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        _model = _model.to(device)
        _model.eval()
        print("✅ Model loaded")
    return _model

def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(image: Image.Image):
    model = get_model()
    img = image.convert("RGB")
    orig_size = img.size
    input_tensor = preprocess(img)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    pred = torch.argmax(output, dim=0).cpu().numpy()

    # Class 8 = building in COCO
    building_mask = (pred == 8).astype(np.uint8)
    building_pixels = int(building_mask.sum())
    total_pixels = building_mask.size
    urban_pct = int((building_pixels / total_pixels) * 100)

    # Create mask image
    mask_img = Image.fromarray(building_mask * 255).resize(orig_size)

    return {
        "mask": mask_img,
        "building_pixels": building_pixels,
        "total_pixels": total_pixels,
        "urban_percentage": urban_pct,
    }
