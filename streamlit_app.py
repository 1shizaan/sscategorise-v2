# streamlit_app.py — SSCategorise v2
# Enhanced: multi-image upload, grid layout, results dashboard
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from PIL import Image
import tempfile, time
from config import CATEGORIES
from ocr.extract import extract_text
from ocr.clean import clean_text
from vision.model import predict_category
from fusion.classifier import predict as fusion_predict

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title='SSCategorise v2',
    page_icon='📂',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* App background */
.stApp { background: #0A0F1E; }

/* Top header bar */
.app-header {
    background: linear-gradient(135deg, #0F1A30 0%, #1A2744 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.app-title { color: #FFFFFF; font-size: 1.6rem; font-weight: 700; margin: 0; }
.app-subtitle { color: #64748B; font-size: 0.85rem; margin: 0; margin-top: 2px; }
.version-badge {
    background: #1E3A5F;
    color: #38BDF8;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
}

/* Upload zone */
.upload-zone {
    background: #0F1A30;
    border: 2px dashed #1E3A5F;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: all 0.2s;
    margin-bottom: 1rem;
}
.upload-zone:hover { border-color: #38BDF8; }

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stat-card {
    background: #0F1A30;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.stat-value { color: #FFFFFF; font-size: 1.5rem; font-weight: 700; line-height: 1; }
.stat-label { color: #64748B; font-size: 0.72rem; font-weight: 500; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }

/* Image result card */
.img-card {
    background: #0F1A30;
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.img-card:hover { border-color: #2A5280; }
.img-filename {
    color: #94A3B8;
    font-size: 0.72rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.category-pill {
    display: inline-block;
    background: #1A3355;
    color: #38BDF8;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.4rem;
}
.fusion-pill {
    display: inline-block;
    background: #1A3A1A;
    color: #34D399;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.4rem;
}
.confidence-bar-bg {
    background: #1E2A3A;
    border-radius: 4px;
    height: 5px;
    margin-top: 6px;
    overflow: hidden;
}
.confidence-bar-fill {
    background: linear-gradient(90deg, #38BDF8, #34D399);
    height: 100%;
    border-radius: 4px;
}
.ocr-preview {
    color: #475569;
    font-size: 0.68rem;
    line-height: 1.4;
    margin-top: 0.5rem;
    font-style: italic;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Progress section */
.progress-label {
    color: #64748B;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0A0F1E;
    border-right: 1px solid #1E3A5F;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* Category filter buttons */
.stButton > button {
    background: #0F1A30;
    color: #94A3B8;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    width: 100%;
    text-align: left;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #1A3355;
    color: #38BDF8;
    border-color: #38BDF8;
}

/* Section heading */
.section-heading {
    color: #E2E8F0;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E3A5F;
}

/* Match/mismatch badge */
.match-badge {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 10px;
    letter-spacing: 0.05em;
}
.match { background: #064E3B; color: #34D399; }
.mismatch { background: #450A0A; color: #F87171; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #475569;
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.1rem; font-weight: 600; color: #64748B; margin-bottom: 0.5rem; }
.empty-sub { font-size: 0.85rem; }

/* File uploader styling */
[data-testid="stFileUploader"] {
    background: #0F1A30;
    border-radius: 12px;
}
[data-testid="stFileUploader"] section {
    background: #0F1A30;
    border: 2px dashed #1E3A5F;
    border-radius: 12px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #0F1A30;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 1.3rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Override default text */
p, div { color: #E2E8F0; }
h1, h2, h3 { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────
CONFIDENCE_COLORS = {
    "high":   ("#34D399", "#064E3B"),
    "medium": ("#FBBF24", "#451A03"),
    "low":    ("#F87171", "#450A0A"),
}

def conf_level(conf):
    if conf >= 0.70: return "high"
    if conf >= 0.45: return "medium"
    return "low"

def process_image(uploaded_file):
    """Process a single uploaded file. Returns result dict."""
    img = Image.open(uploaded_file).convert('RGB')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name, quality=95)
        tmp_path = tmp.name
    try:
        raw_text   = extract_text(tmp_path)
        clean      = clean_text(raw_text)
        img_cat, img_conf = predict_category(tmp_path, CATEGORIES)
        fusion_cat = fusion_predict(tmp_path, clean)
    finally:
        os.unlink(tmp_path)
    return {
        'filename':   uploaded_file.name,
        'image':      img,
        'ocr_text':   clean,
        'img_cat':    img_cat,
        'img_conf':   img_conf,
        'fusion_cat': fusion_cat,
        'match':      img_cat == fusion_cat,
    }


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-heading">⚙️ Settings</p>', unsafe_allow_html=True)

    cols_per_row = st.select_slider(
        'Columns per row',
        options=[2, 3, 4, 5],
        value=3,
    )

    show_ocr = st.toggle('Show OCR text preview', value=True)
    show_conf_bar = st.toggle('Show confidence bar', value=True)

    st.markdown('<br><p class="section-heading">🔍 Filter by Category</p>', unsafe_allow_html=True)

    filter_cat = st.selectbox(
        'Show only',
        ['All categories'] + sorted(CATEGORIES),
        label_visibility='collapsed'
    )

    st.markdown('<br><p class="section-heading">📊 Model Info</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748B; line-height:1.8">
    🟣 <b style="color:#A78BFA">Vision</b> EfficientNet-B3<br>
    🟢 <b style="color:#34D399">Fusion</b> Image + OCR<br>
    📐 Features: 1536 + 1000<br>
    🏷️ Categories: 20<br>
    </div>
    """, unsafe_allow_html=True)

    # Accuracy chart if available
    chart_path = os.path.join(os.path.dirname(__file__), 'fusion', 'accuracy_comparison.png')
    if os.path.exists(chart_path):
        st.markdown('<br><p class="section-heading">📈 Accuracy</p>', unsafe_allow_html=True)
        st.image(chart_path, use_column_width=True)


# ── MAIN HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <p class="app-title">📂 SSCategorise</p>
    <p class="app-subtitle">Multimodal AI — Image + OCR text fusion categorisation</p>
  </div>
  <span class="version-badge">v2.0</span>
</div>
""", unsafe_allow_html=True)


# ── UPLOAD SECTION ────────────────────────────────────────────────
col_upload, col_info = st.columns([3, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        'Upload images to categorise',
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        help='Upload up to 50 images at once. Supported: JPG, PNG, WEBP',
        label_visibility='collapsed',
    )

with col_info:
    st.markdown("""
    <div style="background:#0F1A30; border:1px solid #1E3A5F; border-radius:12px;
                padding:1rem; font-size:0.78rem; color:#64748B; line-height:1.9">
    📁 <b style="color:#94A3B8">Max 50 images</b> at once<br>
    📏 Up to <b style="color:#94A3B8">10MB</b> per file<br>
    🖼️ JPG, PNG, WEBP<br>
    ⚡ ~2-3 sec per image
    </div>
    """, unsafe_allow_html=True)


# ── PROCESS + DISPLAY ─────────────────────────────────────────────
if uploaded_files:
    MAX_FILES = 50
    MAX_SIZE_MB = 10

    # Validate files
    valid_files = []
    skipped = []
    for f in uploaded_files[:MAX_FILES]:
        size_mb = f.size / 1_000_000
        if size_mb > MAX_SIZE_MB:
            skipped.append(f'{f.name} ({size_mb:.1f}MB — too large)')
        else:
            valid_files.append(f)

    if skipped:
        st.warning(f'⚠️ Skipped {len(skipped)} file(s) over 10MB: {", ".join(skipped[:3])}')

    if len(uploaded_files) > MAX_FILES:
        st.info(f'ℹ️ Processing first {MAX_FILES} of {len(uploaded_files)} files.')

    # Process with progress bar
    results = []
    if valid_files:
        progress_placeholder = st.empty()
        status_placeholder   = st.empty()

        with progress_placeholder.container():
            progress_bar = st.progress(0)
            status_text  = st.empty()

        t_start = time.time()
        for i, f in enumerate(valid_files):
            status_text.markdown(
                f'<p class="progress-label">Processing <b style="color:#E2E8F0">{f.name}</b> '
                f'({i+1}/{len(valid_files)})</p>',
                unsafe_allow_html=True
            )
            try:
                result = process_image(f)
                results.append(result)
            except Exception as e:
                results.append({
                    'filename': f.name, 'image': None,
                    'ocr_text': '', 'img_cat': 'Error',
                    'img_conf': 0.0, 'fusion_cat': 'Error',
                    'match': False, 'error': str(e)
                })
            progress_bar.progress((i + 1) / len(valid_files))

        elapsed = time.time() - t_start
        progress_placeholder.empty()
        status_placeholder.empty()

        # ── STATS BAR ─────────────────────────────────────────────
        total     = len(results)
        matched   = sum(1 for r in results if r['match'])
        with_text = sum(1 for r in results if r.get('ocr_text', '').strip())
        cats_found = len(set(r['fusion_cat'] for r in results if r['fusion_cat'] != 'Error'))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Images', total)
        c2.metric('Categories Found', cats_found)
        c3.metric('With Text (OCR)', with_text)
        c4.metric('Vision = Fusion', f'{matched}/{total}')
        c5.metric('Time', f'{elapsed:.1f}s')

        st.markdown('<br>', unsafe_allow_html=True)

        # ── FILTER ────────────────────────────────────────────────
        display_results = results
        if filter_cat != 'All categories':
            display_results = [r for r in results if r['fusion_cat'] == filter_cat]
            if not display_results:
                st.info(f'No images categorised as **{filter_cat}**.')

        # ── CATEGORY SUMMARY ──────────────────────────────────────
        if len(results) > 3:
            with st.expander('📊 Category Breakdown', expanded=False):
                from collections import Counter
                cat_counts = Counter(r['fusion_cat'] for r in results)
                cols = st.columns(4)
                for i, (cat, count) in enumerate(sorted(cat_counts.items(), key=lambda x: -x[1])):
                    with cols[i % 4]:
                        pct = count / total * 100
                        st.markdown(f"""
                        <div style="background:#0F1A30; border:1px solid #1E3A5F;
                                    border-radius:10px; padding:0.7rem; margin-bottom:0.5rem">
                          <div style="font-size:0.72rem; color:#64748B; margin-bottom:4px">{cat}</div>
                          <div style="font-size:1.2rem; font-weight:700; color:#FFFFFF">{count}</div>
                          <div style="background:#1E2A3A; border-radius:3px; height:4px; margin-top:6px">
                            <div style="background:#38BDF8; width:{pct}%; height:100%; border-radius:3px"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── IMAGE GRID ────────────────────────────────────────────
        st.markdown('<p class="section-heading">Results</p>', unsafe_allow_html=True)

        cols = st.columns(cols_per_row)
        for i, result in enumerate(display_results):
            with cols[i % cols_per_row]:
                # Image thumbnail
                if result.get('image'):
                    st.image(result['image'], use_column_width=True)
                else:
                    st.markdown(
                        '<div style="background:#1A1A2E; border-radius:8px; '
                        'height:120px; display:flex; align-items:center; '
                        'justify-content:center; color:#475569">⚠️ Error</div>',
                        unsafe_allow_html=True
                    )

                # Filename
                st.markdown(
                    f'<div class="img-filename" title="{result["filename"]}">'
                    f'{result["filename"]}</div>',
                    unsafe_allow_html=True
                )

                # Predictions
                img_conf = result['img_conf']
                conf_color, conf_bg = CONFIDENCE_COLORS[conf_level(img_conf)]
                match_cls  = 'match' if result['match'] else 'mismatch'
                match_text = '✓ AGREE' if result['match'] else '≠ DIFFER'

                st.markdown(f"""
                <div style="margin-bottom:0.3rem">
                  <span style="font-size:0.65rem; color:#64748B; font-weight:600;
                               text-transform:uppercase; letter-spacing:0.05em">Vision</span>
                  <span class="category-pill">{result['img_cat']}</span>
                  <span style="font-size:0.7rem; color:{conf_color};
                               font-weight:600; float:right">{img_conf:.0%}</span>
                </div>
                """, unsafe_allow_html=True)

                if show_conf_bar:
                    st.markdown(f"""
                    <div class="confidence-bar-bg">
                      <div class="confidence-bar-fill" style="width:{img_conf*100:.0f}%"></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:0.4rem; margin-bottom:0.3rem">
                  <span style="font-size:0.65rem; color:#64748B; font-weight:600;
                               text-transform:uppercase; letter-spacing:0.05em">Fusion</span>
                  <span class="fusion-pill">{result['fusion_cat']}</span>
                  <span class="match-badge {match_cls}" style="float:right">{match_text}</span>
                </div>
                """, unsafe_allow_html=True)

                # OCR preview
                if show_ocr and result.get('ocr_text', '').strip():
                    preview = result['ocr_text'][:80]
                    st.markdown(
                        f'<div class="ocr-preview">"{preview}..."</div>',
                        unsafe_allow_html=True
                    )
                elif show_ocr:
                    st.markdown(
                        '<div class="ocr-preview">No text detected</div>',
                        unsafe_allow_html=True
                    )

                # Expander for full OCR
                if result.get('ocr_text', '').strip():
                    with st.expander('Full OCR text'):
                        st.text(result['ocr_text'])

                st.markdown('<hr style="border-color:#1E3A5F; margin:0.5rem 0">', unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">📂</div>
      <div class="empty-title">No images uploaded yet</div>
      <div class="empty-sub">
        Drag and drop images above, or click Browse.<br>
        Upload up to 50 images at once for batch categorisation.
      </div>
    </div>
    """, unsafe_allow_html=True)
