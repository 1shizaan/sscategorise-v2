# vision/model.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
from config import VISION_MODEL_PATH, NUM_CLASSES, IMAGE_SIZE

transform = T.Compose([
    T.Resize(320),
    T.CenterCrop(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_model_cache = None

def load_model():
    """Load EfficientNet-B0 with trained weights."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    if os.path.exists(VISION_MODEL_PATH):
        m.load_state_dict(torch.load(VISION_MODEL_PATH, map_location='cpu'))
        print(f'Loaded weights from {VISION_MODEL_PATH}')
    else:
        print('WARNING: No saved weights found. Using ImageNet init only.')
    m.eval()
    _model_cache = m
    return m

def get_image_features(image_path: str, model=None) -> np.ndarray:
    """Returns 1280-dim feature vector. CONTRACT function for Fusion Engineer."""
    if model is None:
        model = load_model()
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f'Could not open {image_path}: {e}')
        return np.zeros(1536)
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feats = model.features(tensor)
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1)
        feats = feats.squeeze().numpy()
    return feats

def predict_category(image_path: str, categories: list, model=None) -> tuple:
    """Returns (top_category_str, confidence_float) for a single image."""
    if model is None:
        model = load_model()
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1).squeeze()
    idx = probs.argmax().item()
    return categories[idx], probs[idx].item()
