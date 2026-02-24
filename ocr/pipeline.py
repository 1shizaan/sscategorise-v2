import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
import pandas as pd 
from config import DATASET_PATH, OCR_OUTPUT_PATH, PROCESSED_CSV 
from ocr.extract import extract_text 
from ocr.clean import clean_text 
 
def build_dataset_csv(): 
    """ 
    Walks categorized_images/, runs OCR + cleaning on each image, 
    and writes processed_dataset.csv with columns: image_path, ocr_text, label 
    """ 
    rows = [] 
    for category in sorted(os.listdir(DATASET_PATH)): 
        cat_path = os.path.join(DATASET_PATH, category) 
        if not os.path.isdir(cat_path): continue 
        for fname in os.listdir(cat_path): 
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue 
            img_path = os.path.join(cat_path, fname) 
            # Try loading from cached OCR txt first 
            txt_name = os.path.splitext(fname)[0] + '.txt' 
            txt_path = os.path.join(OCR_OUTPUT_PATH, txt_name) 
            if os.path.exists(txt_path): 
                with open(txt_path, 'r', encoding='utf-8') as f: 
                    raw = f.read() 
            else: 
                raw = extract_text(img_path)  # fallback: run OCR now 
            cleaned = clean_text(raw) 
            rows.append({'image_path': img_path, 'ocr_text': cleaned, 'label': 
category}) 
    df = pd.DataFrame(rows) 
    df.to_csv(PROCESSED_CSV, index=False) 
    print(f'Saved {len(df)} rows to {PROCESSED_CSV}') 
    print(df['label'].value_counts()) 
    return df 
 
if __name__ == '__main__': 
    build_dataset_csv() 
