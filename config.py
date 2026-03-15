# config.py — place in ROOT of project
import os

# ── Categories ────────────────────────────────────────────
CATEGORIES = [
    "Automotive",
    "Books and Magazines",
    "E-commerce",
    "Education",
    "Entertainment",
    "Finance",
    "Food and Dining",
    "Government and Public Services",
    "Motivational Posters",
    "Nature",
    "News",
    "Productivity Tools",
    "Quotes (Plain Cards)",
    "Receipts",
    "Recipes",
    "Resources",
    "Ronaldo (Celebrities)",
    "Social Media Screens",
    "Sports",
    "Technology",
]
NUM_CLASSES = len(CATEGORIES)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
DATASET_PATH      = os.path.join(DATA_DIR, "categorized_images")
OCR_OUTPUT_PATH   = os.path.join(DATA_DIR, "ocr_outputs")
PROCESSED_CSV     = os.path.join(DATA_DIR, "processed_dataset.csv")

VISION_DIR        = os.path.join(BASE_DIR, "vision")
VISION_MODEL_PATH = os.path.join(VISION_DIR, "saved_models", "efficientnet_b3.pth")

FUSION_DIR        = os.path.join(BASE_DIR, "fusion")
FUSION_MODEL_PATH = os.path.join(FUSION_DIR, "saved_models", "fusion_model.pkl")
FUSION_TFIDF_PATH = os.path.join(FUSION_DIR, "saved_models", "tfidf.pkl")

# ── Model settings ────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16

# ── Create required dirs automatically ───────────────────
for d in [DATA_DIR, DATASET_PATH, OCR_OUTPUT_PATH,
          os.path.join(VISION_DIR, "saved_models"),
          os.path.join(FUSION_DIR, "saved_models")]:
    os.makedirs(d, exist_ok=True)
