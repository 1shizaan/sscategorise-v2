# ocr/extract.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytesseract
from PIL import Image
from config import DATASET_PATH, OCR_OUTPUT_PATH

# Windows only: uncomment and set path to tesseract.exe if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path: str) -> str:
    """Extract raw text from a single image using Tesseract."""
    try:
        img = Image.open(image_path).convert('RGB')
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f'OCR failed for {image_path}: {e}')
        return ''

def run_ocr_on_dataset():
    """Run OCR on every image in the dataset. Saves .txt files to OCR_OUTPUT_PATH."""
    processed = 0
    for category in os.listdir(DATASET_PATH):
        cat_path = os.path.join(DATASET_PATH, category)
        if not os.path.isdir(cat_path): continue
        for fname in os.listdir(cat_path):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            img_path = os.path.join(cat_path, fname)
            text = extract_text(img_path)
            out_name = os.path.splitext(fname)[0] + '.txt'
            out_path = os.path.join(OCR_OUTPUT_PATH, out_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text)
            processed += 1
            if processed % 50 == 0:
                print(f'  Processed {processed} images...')
    print(f'OCR complete. {processed} files written to {OCR_OUTPUT_PATH}')

if __name__ == '__main__':
    run_ocr_on_dataset()
