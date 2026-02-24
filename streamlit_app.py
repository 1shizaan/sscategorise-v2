# streamlit_app.py — root of project 
import sys, os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) 
import streamlit as st 
from PIL import Image 
import tempfile 
from config import CATEGORIES 
from ocr.extract import extract_text 
from ocr.clean import clean_text 
from vision.model import predict_category 
from fusion.classifier import predict as fusion_predict 
 
st.set_page_config(page_title='SS Categorise v2', page_icon='📂') 
st.title('📂 SS Categorise v2 — Multimodal AI') 
st.caption('Image + OCR text fusion for smarter categorization') 
 
uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png']) 
 
if uploaded: 
    img = Image.open(uploaded).convert('RGB') 
    st.image(img, caption=uploaded.name, use_column_width=True) 
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp: 
        img.save(tmp.name) 
        tmp_path = tmp.name 
    with st.spinner('Running OCR...'): 
        raw_text   = extract_text(tmp_path) 
        clean      = clean_text(raw_text) 
    with st.expander('Extracted Text (OCR)'): 
        st.text(clean if clean else 'No text detected in this image.') 
    with st.spinner('Running classification...'): 
        img_cat, img_conf   = predict_category(tmp_path, CATEGORIES) 
        fusion_cat          = fusion_predict(tmp_path, clean) 
    col1, col2 = st.columns(2) 
    col1.metric('Image-Only Prediction', img_cat, f'{img_conf:.0%} confidence') 
    col2.metric('Multimodal Prediction (v2)', fusion_cat) 
    os.unlink(tmp_path) 
