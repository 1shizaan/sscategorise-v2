# fusion/classifier.py 
import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
import pickle, numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from config import FUSION_MODEL_PATH, FUSION_TFIDF_PATH 
from fusion.feature_builder import build_features 
 
def train_fusion_classifier(): 
    """Train and save the fusion classifier. Returns test accuracy.""" 
    X, y, le = build_features(fit_tfidf=True) 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=42, stratify=y 
    ) 
    clf = LogisticRegression(max_iter=1000, C=1.0) 
    clf.fit(X_train, y_train) 
    acc = accuracy_score(y_test, clf.predict(X_test)) 
    print(f'Fusion Classifier Accuracy: {acc*100:.2f}%') 
    with open(FUSION_MODEL_PATH, 'wb') as f: 
        pickle.dump({'model': clf, 'label_encoder': le}, f) 
    print(f'Saved fusion model to {FUSION_MODEL_PATH}') 
    return acc 
 
def predict(image_path: str, ocr_text: str) -> str: 
    """Main prediction function — called by streamlit_app.py""" 
    from vision.model import get_image_features 
    with open(FUSION_MODEL_PATH, 'rb') as f: 
        bundle = pickle.load(f) 
    with open(FUSION_TFIDF_PATH, 'rb') as f: 
        tfidf = pickle.load(f) 
    img_feat  = get_image_features(image_path).reshape(1, -1) 
    text_feat = tfidf.transform([ocr_text]).toarray() 
    X = np.hstack([img_feat, text_feat]) 
    pred_idx  = bundle['model'].predict(X)[0] 
    return bundle['label_encoder'].inverse_transform([pred_idx])[0] 
 
if __name__ == '__main__': 
    train_fusion_classifier() 
