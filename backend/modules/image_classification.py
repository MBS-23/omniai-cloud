# backend/modules/image_classification.py

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

# Load model + official preprocessing
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

# Load ImageNet labels
IMAGENET_LABELS = weights.meta["categories"]

def classify_image(image_path: str):
    """
    Classify an image using ResNet50 (ImageNet)
    """

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_idx = torch.max(probs, dim=0)

    return {
        "label": IMAGENET_LABELS[top_idx.item()],
        "confidence": round(top_prob.item() * 100, 2)
    }
