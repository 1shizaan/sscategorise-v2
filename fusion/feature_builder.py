# fusion/feature_builder.py 
import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
import numpy as np 
import pandas as pd 
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder 
from config import PROCESSED_CSV, FUSION_TFIDF_PATH 
from vision.model import get_image_features 
 
def build_features(df: pd.DataFrame = None, fit_tfidf: bool = True): 
    """ 
    Returns X (fused features), y (encoded labels), label_encoder. 
    If fit_tfidf=True: fits a new TF-IDF and saves it. Else: loads saved one. 
    """ 
    if df is None: 
        df = pd.read_csv(PROCESSED_CSV) 
    df['ocr_text'] = df['ocr_text'].fillna('') 
    # 1. Image features 
    print('Extracting image features... (this takes a while)') 
    image_feats = np.array([get_image_features(p) for p in df['image_path']]) 
    # 2. TF-IDF text features 
    if fit_tfidf: 
        tfidf = TfidfVectorizer(max_features=500, stop_words='english') 
        text_feats = tfidf.fit_transform(df['ocr_text']).toarray() 
        with open(FUSION_TFIDF_PATH, 'wb') as f: 
            pickle.dump(tfidf, f) 
        print(f'Saved TF-IDF to {FUSION_TFIDF_PATH}') 
    else: 
        with open(FUSION_TFIDF_PATH, 'rb') as f: 
            tfidf = pickle.load(f) 
        text_feats = tfidf.transform(df['ocr_text']).toarray() 
    # 3. Fuse by concatenation 
    X = np.hstack([image_feats, text_feats])  # shape: (N, 1780) 
    le = LabelEncoder() 
    y = le.fit_transform(df['label']) 
    print(f'Feature matrix shape: {X.shape}') 
    return X, y, le
