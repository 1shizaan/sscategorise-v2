# main.py — quick sanity check runner
# Run this to verify all modules are importable before the team starts coding
# Usage: python3 main.py

import sys, os

print("=" * 50)
print("SS Categorise v2 — Startup Check")
print("=" * 50)

# 1. Config
try:
    from config import CATEGORIES, NUM_CLASSES, DATASET_PATH
    print(f"✅ config.py          — {NUM_CLASSES} classes loaded")
except Exception as e:
    print(f"❌ config.py          — {e}")

# 2. Dataset check
try:
    folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    total_images = sum(
        len(os.listdir(os.path.join(DATASET_PATH, f))) for f in folders
    )
    print(f"✅ categorized_images — {len(folders)} folders, {total_images} images")
except Exception as e:
    print(f"❌ categorized_images — {e}")

# 3. Vision module
try:
    from vision.model import load_model, get_image_features
    print(f"✅ vision/model.py    — imports OK")
except Exception as e:
    print(f"❌ vision/model.py    — {e}")

# 4. OCR module
try:
    from ocr.extract import extract_text
    from ocr.clean import clean_text
    print(f"✅ ocr/               — imports OK")
except Exception as e:
    print(f"❌ ocr/               — {e}")

# 5. Fusion module
try:
    from fusion.feature_builder import build_features
    from fusion.classifier import predict
    print(f"✅ fusion/            — imports OK")
except Exception as e:
    print(f"❌ fusion/            — {e}")

print("=" * 50)
print("Run 'streamlit run streamlit_app.py' to launch the app")
print("=" * 50)
