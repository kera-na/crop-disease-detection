import hashlib
import hmac
import json
import os
import re
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from modules.model_loader import load_deployment
from modules.predictor import predict_image, preprocess_image
from modules.utils import get_severity, models_agree
from modules.disease_db import DISEASES
from modules.pdf_report import create_pdf_report
from modules.gradcam import find_last_conv_layer, make_gradcam_heatmap
from modules.ai_assistant import ask_farmer_assistant
from modules.experts import (
    SPECIALIZATIONS, add_expert, delete_expert, geocode, haversine_km,
    db_info, import_experts, init_db, list_experts, set_verified,
)


APP_NAME = "Crop Disease Detection System"
DEFAULT_ADMIN_CODE = "2026"  # used only when ADMIN_PASSWORD is not set
CROPS = ["Cashew", "Cassava", "Maize", "Tomato"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLOR THEMES (light / dark mode and palette, selectable from the sidebar)
# ============================================================

THEMES = {
    "Tropical Agriculture": {
        "Light": dict(
            dark=False, bg="#FFFBEF", sidebar="#E6F4EF", surface="#FFFFFF", surface2="#F1F7E4",
            border="#D3E3CC", text="#0B3D36", muted="#4F7A72", accent="#00695C",
            accent_hover="#004D43", on_accent="#FFFFFF", warn="#F9A825",
            chart=["#00695C", "#8BC34A", "#F9A825", "#26A69A", "#E57F5B", "#7E57C2"],
            template="plotly_white", map_style="carto-positron",
        ),
        "Dark": dict(
            dark=True, bg="#0C3A33", sidebar="#082A25", surface="#104840", surface2="#165A50",
            border="#2A7A6C", text="#FFF8E1", muted="#B5D6CD", accent="#8BC34A",
            accent_hover="#76A83C", on_accent="#0B1F08", warn="#F9A825",
            chart=["#8BC34A", "#26A69A", "#F9A825", "#E57F5B", "#80CBC4", "#C5E1A5"],
            template="plotly_dark", map_style="carto-darkmatter",
        ),
    },
    "Earth & Harvest": {
        "Light": dict(
            dark=False, bg="#F4EBDD", sidebar="#EADFCB", surface="#FBF6EC", surface2="#EFE4D0",
            border="#D9C9AA", text="#2B3318", muted="#6B6B4A", accent="#556B2F",
            accent_hover="#44571F", on_accent="#FFFFFF", warn="#B87333",
            chart=["#556B2F", "#B87333", "#A3A847", "#C49A6C", "#7A5C3A", "#3F7C85"],
            template="plotly_white", map_style="open-street-map",
        ),
        "Dark": dict(
            dark=True, bg="#16190F", sidebar="#11140B", surface="#1E2214", surface2="#272C1A",
            border="#3A4128", text="#F4EBDD", muted="#B5AE94", accent="#A3A847",
            accent_hover="#8C9139", on_accent="#1A1C06", warn="#B87333",
            chart=["#A3A847", "#B87333", "#C49A6C", "#8FAE5B", "#D98C4A", "#7FA08B"],
            template="plotly_dark", map_style="carto-darkmatter",
        ),
    },
    "Forest": {
        "Light": dict(
            dark=False, bg="#F5F7F6", sidebar="#FFFFFF", surface="#FFFFFF", surface2="#EEF3F0",
            border="#D9E2DD", text="#14231C", muted="#5B6F65", accent="#1F7A4D",
            accent_hover="#17633E", on_accent="#FFFFFF", warn="#B7791F",
            chart=["#1F7A4D", "#2F6FDE", "#D69E2E", "#7C5CD6", "#D9534F", "#14A3A3"],
            template="plotly_white", map_style="carto-positron",
        ),
        "Dark": dict(
            dark=True, bg="#0d1814", sidebar="#0a130f", surface="#14231d", surface2="#1a2e26",
            border="#284237", text="#e9f1ed", muted="#9fb5aa", accent="#4cc38a",
            accent_hover="#3aa974", on_accent="#06140d", warn="#e5b454",
            chart=["#4CC38A", "#5BA8D6", "#E5B454", "#9B8AE0", "#E07A6B", "#7FD1C7"],
            template="plotly_dark", map_style="carto-darkmatter",
        ),
    },
}
DEFAULT_FAMILY = "Tropical Agriculture"
MODES = ["Light", "Dark"]


def native_theme_type():
    """Streamlit's own light/dark setting (None on older Streamlit versions)."""
    try:
        return st.context.theme.type
    except Exception:
        return None


native_type = native_theme_type()
default_mode = "Dark" if native_type == "dark" else "Light"

theme_family = st.session_state.get("theme_family", DEFAULT_FAMILY)
if theme_family not in THEMES:
    theme_family = DEFAULT_FAMILY
theme_mode = st.session_state.get("theme_mode", default_mode)
if theme_mode not in MODES:
    theme_mode = default_mode
T = THEMES[theme_family][theme_mode]
CHART_COLORS = T["chart"]

theme_vars = f"""
        color-scheme: {"dark" if T["dark"] else "light"};
        --bg: {T["bg"]}; --sidebar: {T["sidebar"]}; --surface: {T["surface"]};
        --surface-2: {T["surface2"]}; --border: {T["border"]}; --text: {T["text"]};
        --muted: {T["muted"]}; --accent: {T["accent"]}; --accent-hover: {T["accent_hover"]};
        --on-accent: {T["on_accent"]}; --warn: {T["warn"]};
        --shadow: {"0 8px 24px rgba(0,0,0,.32)" if T["dark"] else "0 1px 3px rgba(20,35,28,.10)"};
"""

# Streamlit draws tables on a canvas using its own theme. If the chosen mode
# differs from Streamlit's theme, invert the table so it matches the page.
theme_extra_css = ""
if native_type in ("light", "dark") and native_type != theme_mode.lower():
    theme_extra_css = (
        '[data-testid="stDataFrame"] { filter: invert(1) hue-rotate(180deg); }'
    )



# ============================================================
# STYLING (single consistent dark theme)
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
        /*VARS*/
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: var(--bg) !important; color: var(--text); }
    [data-testid="stMain"] { background: transparent !important; }
    [data-testid="stMainBlockContainer"] { padding-top: 1.6rem; max-width: 1200px; }
    header[data-testid="stHeader"] { background: transparent; }

    h1, h2, h3, h4 { color: var(--text) !important; font-weight: 700 !important; letter-spacing: -0.01em; }
    p, li, label, [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p, [data-testid="stRadio"] label { color: var(--text) !important; }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: var(--sidebar) !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important;
    }
    [data-testid="stSidebar"] hr { border-color: var(--border) !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 8px 12px; border-radius: 8px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: var(--surface); }
    .brand { display:flex; align-items:center; gap:12px; padding: 6px 2px 18px 2px; }
    .brand-mark { background: var(--accent); color:var(--on-accent); width:38px; height:38px; border-radius:10px;
        display:flex; align-items:center; justify-content:center; font-size:1.2rem; }
    .brand-name { font-weight:700; font-size:1rem; line-height:1.2; color: var(--text); }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, var(--surface-2), var(--surface));
        border: 1px solid var(--border); border-radius: 16px;
        padding: 26px 30px; margin-bottom: 24px;
    }
    .page-title { font-size: clamp(1.6rem, 3vw, 2.2rem); font-weight: 700; color: var(--text); margin: 0 0 6px 0; }
    .page-subtitle { color: var(--muted); font-size: .98rem; margin: 0; max-width: 700px; }

    /* Section headings */
    .section-kicker { color: var(--accent); font-size: .74rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 4px; }
    .section-heading { font-size: 1.35rem; font-weight: 700; color: var(--text); margin: 0 0 4px 0; }
    .section-subtitle { color: var(--muted); margin: 0 0 12px 0; font-size: .92rem; }
    .section-divider { height: 1px; background: var(--border); margin: 10px 0 18px 0; }

    /* Cards / containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface); border: 1px solid var(--border) !important; border-radius: 14px;
    }
    .feature-card { border-left: 3px solid var(--accent); padding: 6px 0 6px 14px; }
    .feature-icon { font-size: 1.3rem; margin-bottom: 6px; }
    .feature-title { font-weight: 600; color: var(--text); margin-bottom: 4px; }
    .feature-copy { color: var(--muted); font-size: .86rem; line-height: 1.45; }

    .result-panel { background: var(--surface-2); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 12px; padding: 18px 20px; }
    .result-disease { color: var(--text); font-weight: 700; font-size: 1.4rem; line-height: 1.25; margin: 4px 0 10px 0; overflow-wrap: anywhere; }
    .soft-note { color: var(--muted); font-size: .86rem; line-height: 1.5; }

    /* Buttons */
    .stButton > button, .stDownloadButton > button, .stLinkButton > a {
        border-radius: 8px; border: 1px solid transparent;
        background: var(--accent) !important; color: var(--on-accent) !important;
        font-weight: 600; padding: .55rem 1rem; transition: .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover { background: var(--accent-hover) !important; }
    .stButton > button[kind="secondary"] { background: transparent !important; color: var(--text) !important; border: 1px solid var(--border); }

    /* Metrics */
    [data-testid="stMetric"] { background: var(--surface-2); border: 1px solid var(--border); padding: 12px 16px; border-radius: 12px; }
    [data-testid="stMetricLabel"] p { color: var(--muted) !important; }
    [data-testid="stMetricValue"] { color: var(--text) !important; }

    /* Inputs */
    [data-testid="stFileUploaderDropzone"] { background: var(--surface-2) !important; border: 1px dashed var(--accent) !important; border-radius: 10px; }
    [data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
    [data-testid="stFileUploaderDropzone"] button { background: var(--accent) !important; color: var(--on-accent) !important; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: var(--surface-2) !important; color: var(--text) !important; border-color: var(--border) !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="select"] [role="option"] { background: var(--surface-2) !important; color: var(--text) !important; }
    [data-testid="stAlert"] { background: var(--surface-2) !important; color: var(--text) !important; border: 1px solid var(--border); border-radius: 10px; }
    [data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

    /* Tabs */
    div[data-testid="stTabs"] [role="tablist"] { gap: 4px; border-bottom: 1px solid var(--border); }
    div[data-testid="stTabs"] button { color: var(--muted) !important; font-weight: 600; padding: 10px 16px; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--accent) !important; }

    hr { border-color: var(--border) !important; }

    /* Polish */
    .hero {
        background:
            radial-gradient(circle at 0% 0%, color-mix(in srgb, var(--accent) 16%, transparent), transparent 55%),
            var(--surface) !important;
        box-shadow: var(--shadow);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { box-shadow: var(--shadow); }
    [data-testid="stMetricLabel"] p { font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent); }
    [data-testid="stProgress"] > div > div > div > div { background: var(--accent) !important; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    /*EXTRA*/
    .footer { text-align: center; color: var(--muted); font-size: .8rem; padding: 24px 0 8px 0; }
    </style>
    """.replace("/*VARS*/", theme_vars).replace("/*EXTRA*/", theme_extra_css),
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def cached_load_deployment(model_folder):
    """Load a deployment model once per folder path."""
    return load_deployment(model_folder)


def load_model_safely(model_folder, label):
    if not model_folder or not os.path.isdir(model_folder):
        return None, None, f"{label} folder not found: {model_folder}"
    try:
        model, classes, _config = cached_load_deployment(model_folder)
        return model, classes, None
    except Exception as exc:
        return None, None, f"{label} could not load: {exc}"


def get_image_hash(pil_image):
    return hashlib.sha256(pil_image.tobytes()).hexdigest()


def add_history_record(image_name, disease, confidence, model_name, image_hash):
    """Avoid adding the same image repeatedly during Streamlit reruns."""
    if image_hash in {item.get("Image hash") for item in st.session_state.history}:
        return
    st.session_state.history.insert(
        0,
        {
            "Date and time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Image": image_name,
            "Disease": disease,
            "Confidence (%)": round(confidence * 100, 2),
            "Model": model_name,
            "Image hash": image_hash,
        },
    )


def render_section_heading(kicker, heading, subtitle=None):
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-heading">{heading}</div>
        <div class="section-subtitle">{subtitle or ""}</div>
        <div class="section-divider"></div>
        """,
        unsafe_allow_html=True,
    )


def render_model_result(title, result):
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown(f"**{title}**")
    st.markdown(f'<div class="result-disease">{result["class"]}</div>', unsafe_allow_html=True)
    st.metric("Confidence", f'{result["confidence"] * 100:.2f}%')
    st.markdown("</div>", unsafe_allow_html=True)


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Blend a Grad-CAM heatmap over the original image."""
    hm = np.asarray(heatmap, dtype=float).squeeze()
    if hm.ndim != 2:
        raise ValueError("Heatmap must be two-dimensional.")
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = np.asarray(
        Image.fromarray(np.uint8(255 * hm)).resize(image.size, Image.BILINEAR)
    ) / 255.0
    colored = np.stack(
        [
            np.clip(1.5 - np.abs(4 * hm - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * hm - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * hm - 1), 0, 1),
        ],
        axis=-1,
    ) * 255
    base = np.asarray(image.convert("RGB"), dtype=float)
    return Image.fromarray(((1 - alpha) * base + alpha * colored).astype(np.uint8))


def styled_bar(df, x, y="Detections", angle=0):
    fig = px.bar(
        df, x=x, y=y, color=x,
        color_discrete_sequence=CHART_COLORS, template=T["template"],
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]), margin=dict(l=8, r=8, t=12, b=8),
        xaxis=dict(title=None, tickangle=angle, gridcolor=T["border"], linecolor=T["border"]),
        yaxis=dict(title=y, gridcolor=T["border"], zerolinecolor=T["border"]),
    )
    return fig


# ---- Gemini assistant ----

def verify_gemini_api_key():
    """Verify the key with a small request and keep it in session memory only."""
    candidate = st.session_state.get("gemini_api_key_entry", "").strip()
    st.session_state.pop("gemini_api_key_entry", None)

    if not candidate:
        st.session_state.gemini_api_status = "Enter your Gemini API key first."
        st.session_state.gemini_api_connected = False
        return

    try:
        ask_farmer_assistant(
            candidate, "API connection test", 0.0,
            "Reply with exactly: API connection successful.",
        )
        st.session_state.gemini_api_key = candidate
        st.session_state.gemini_api_connected = True
        st.session_state.gemini_api_status = None  # nothing to show once connected
    except Exception as exc:
        st.session_state.gemini_api_connected = False
        st.session_state.gemini_api_key = None
        st.session_state.gemini_api_status = f"Connection failed: {exc}"


def disconnect_gemini_api():
    st.session_state.gemini_api_key = None
    st.session_state.gemini_api_connected = False
    st.session_state.gemini_api_status = None
    st.session_state.assistant_answer = None


def submit_farmer_question():
    api_key = st.session_state.get("gemini_api_key")
    question = st.session_state.get("farmer_question_input", "").strip()
    disease = st.session_state.get("assistant_disease_context", "")
    confidence = st.session_state.get("assistant_confidence_context", 0.0)

    try:
        if not api_key or not st.session_state.get("gemini_api_connected", False):
            st.session_state.assistant_error = "Connect and verify your Gemini API key first."
        elif not question:
            st.session_state.assistant_error = "Type your farming question first."
        else:
            st.session_state.assistant_answer = ask_farmer_assistant(
                api_key, disease, confidence, question
            )
            st.session_state.assistant_error = None
    except Exception as exc:
        st.session_state.assistant_error = f"The AI assistant request failed: {exc}"
        if any(t in str(exc).lower() for t in ("api key", "unauthorized", "401", "403", "invalid")):
            st.session_state.gemini_api_connected = False
            st.session_state.gemini_api_key = None
            st.session_state.gemini_api_status = "Authentication failed. Please reconnect your key."


# ============================================================
# SESSION STATE
# ============================================================

for key, default in {
    "history": [],
    "assistant_answer": None,
    "assistant_error": None,
    "gemini_api_key": None,
    "gemini_api_connected": False,
    "gemini_api_status": None,
}.items():
    st.session_state.setdefault(key, default)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand"><div class="brand-mark">🌿</div><div class="brand-name">Crop Disease<br>Detection System</div></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    page = st.radio(
        "Open section",
        ["Disease Detection", "Business Analytics", "Find an Expert", "Prediction History", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Model configuration")
    cnn_path = st.text_input("CNN model folder", value=os.getenv("CNN_MODEL_DIR", "deployment_model/cnn"))
    mobile_path = st.text_input(
        "MobileNetV2 model folder",
        value=os.getenv("MOBILENET_MODEL_DIR", "deployment_model/mobilenetv2"),
    )
    analysis_mode = st.selectbox(
        "Analysis mode", ["Compare Both Models", "CNN", "MobileNetV2"], index=0
    )

    st.markdown("---")
    st.markdown("### Appearance")
    st.radio(
        "Mode", MODES, key="theme_mode", horizontal=True,
        index=MODES.index(default_mode),
    )
    st.selectbox(
        "Color palette", list(THEMES), key="theme_family",
        index=list(THEMES).index(DEFAULT_FAMILY),
    )

    st.markdown("---")
    st.caption("Supported crops")
    st.markdown("\n".join(f"- {c}" for c in CROPS))


# ============================================================
# LOAD MODELS
# ============================================================

cnn_model, cnn_classes, cnn_error = load_model_safely(cnn_path, "CNN")
mobile_model, mobile_classes, mobile_error = load_model_safely(mobile_path, "MobileNetV2")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="page-title">{APP_NAME}</div>
        <div class="page-subtitle">
            Screen crop leaf images for disease, compare deep learning models,
            and review practical treatment guidance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE: DISEASE DETECTION
# ============================================================

if page == "Disease Detection":

    for err, ready in ((cnn_error, "CNN model ready"), (mobile_error, "MobileNetV2 model ready")):
        if err:
            st.sidebar.warning(err)
        else:
            st.sidebar.success(ready)

    render_section_heading(
        "Step 1",
        "Provide a leaf image",
        "Use a clear, well-lit photo of a single leaf with the affected area in focus.",
    )

    image = None
    image_name = ""
    with st.container(border=True):
        input_col, preview_col = st.columns([1.05, 0.95], gap="large")

        with input_col:
            input_type = st.radio("Image source", ["Upload image", "Use camera"], horizontal=True)

            if input_type == "Upload image":
                uploaded_file = st.file_uploader(
                    "Choose a leaf photo", type=["jpg", "jpeg", "png"],
                    help="Supported formats: JPG and PNG.",
                )
                if uploaded_file is not None:
                    try:
                        image = Image.open(uploaded_file).convert("RGB")
                        image_name = uploaded_file.name
                    except Exception as exc:
                        st.error(f"Could not read the uploaded image: {exc}")
            else:
                camera_file = st.camera_input("Take a leaf photo")
                if camera_file is not None:
                    try:
                        image = Image.open(camera_file).convert("RGB")
                        image_name = "Camera capture"
                    except Exception as exc:
                        st.error(f"Could not read the camera image: {exc}")

            st.caption("Avoid strong shadows, blur, and backgrounds that hide the leaf.")

        with preview_col:
            if image is not None:
                st.image(image, caption=image_name, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div style="min-height: 280px; display: flex; flex-direction: column;
                        align-items: center; justify-content: center; background: var(--surface-2);
                        border: 1px dashed var(--border); border-radius: 12px; text-align: center; padding: 24px;">
                        <div style="font-size: 2.6rem;">🍃</div>
                        <div style="font-weight: 600; margin-top: 8px;">Leaf preview</div>
                        <div style="font-size: .85rem; margin-top: 4px; color: var(--muted);">
                            Your image will appear here.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if image is not None:
        image_hash = get_image_hash(image)
        result = selected_model = cnn_result = mobile_result = None

        with st.spinner("Analyzing the leaf..."):
            try:
                if analysis_mode == "CNN":
                    if cnn_model is None:
                        st.error("CNN model is not ready. Check the model folder in the sidebar.")
                    else:
                        result, selected_model = predict_image(image, cnn_model, cnn_classes), "CNN"
                elif analysis_mode == "MobileNetV2":
                    if mobile_model is None:
                        st.error("MobileNetV2 is not ready. Check the model folder in the sidebar.")
                    else:
                        result, selected_model = predict_image(image, mobile_model, mobile_classes), "MobileNetV2"
                else:
                    if cnn_model is None or mobile_model is None:
                        st.error(
                            "Comparison requires both models. Check both model folders "
                            "or select a single available model in the sidebar."
                        )
                    else:
                        cnn_result = predict_image(image, cnn_model, cnn_classes)
                        mobile_result = predict_image(image, mobile_model, mobile_classes)
                        if cnn_result["confidence"] >= mobile_result["confidence"]:
                            result, selected_model = cnn_result, "CNN"
                        else:
                            result, selected_model = mobile_result, "MobileNetV2"
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        if result is not None:
            disease = result["class"]
            confidence = float(result["confidence"])
            severity = get_severity(confidence)
            info = DISEASES.get(disease, {})
            add_history_record(image_name, disease, confidence, selected_model, image_hash)

            st.write("")
            render_section_heading(
                "Step 2",
                "Analysis",
                "Predictions are screening results, not a confirmed laboratory diagnosis.",
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", disease)
            m2.metric("Confidence", f"{confidence * 100:.2f}%")
            m3.metric("Confidence level", severity)
            st.progress(max(0.0, min(confidence, 1.0)))
            st.write("")

            tab_results, tab_guidance, tab_xai, tab_ai = st.tabs(
                ["Results", "Guidance", "Explainability", "AI Agronomist"]
            )

            # ---- Results ----
            with tab_results:
                if cnn_result is not None and mobile_result is not None:
                    left, right = st.columns(2, gap="large")
                    with left:
                        render_model_result("CNN model", cnn_result)
                    with right:
                        render_model_result("MobileNetV2 model", mobile_result)
                    st.write("")
                    if models_agree(cnn_result["class"], mobile_result["class"]):
                        st.success("Both models predict the same disease class.")
                    else:
                        st.warning(
                            "The models disagree. Review the leaf image carefully "
                            "and consider expert confirmation."
                        )
                    st.caption(
                        f"The report and explainability view use the higher-confidence "
                        f"prediction from {selected_model}."
                    )
                else:
                    render_model_result(f"{selected_model} model", result)

                st.write("")
                report_col, note_col = st.columns([1, 2])
                with report_col:
                    try:
                        st.download_button(
                            "⬇️ Download PDF report",
                            data=create_pdf_report(disease, confidence * 100, severity),
                            file_name="crop_disease_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.error(f"PDF report generation failed: {exc}")
                with note_col:
                    st.markdown(
                        '<div class="soft-note">The report contains the prediction, confidence '
                        'score, and confidence level. Keep a copy with your inspection records.</div>',
                        unsafe_allow_html=True,
                    )

            # ---- Guidance ----
            with tab_guidance:
                c1, c2, c3 = st.columns(3, gap="medium")
                for col, heading, field, fallback in [
                    (c1, "🩺 Disease information", "description", "No disease description is available."),
                    (c2, "💊 Treatment", "treatment", "No treatment guidance is available."),
                    (c3, "🛡️ Prevention", "prevention", "No prevention guidance is available."),
                ]:
                    with col:
                        with st.container(border=True):
                            st.markdown(f"#### {heading}")
                            st.write(info.get(field, fallback))

            # ---- Explainability ----
            with tab_xai:
                st.write(
                    "Grad-CAM highlights the image regions that most influenced the "
                    "model. Use it as a visual aid, not as proof of disease."
                )
                try:
                    model_obj = cnn_model if selected_model == "CNN" else mobile_model
                    if model_obj is None:
                        st.warning("The selected model is not available.")
                    else:
                        input_shape = model_obj.input_shape
                        if isinstance(input_shape, list):
                            input_shape = input_shape[0]
                        height, width = input_shape[1], input_shape[2]

                        if height is None or width is None:
                            st.warning("The model input dimensions are dynamic. Grad-CAM needs fixed dimensions.")
                        else:
                            img_array = preprocess_image(image, width, height)
                            last_conv = find_last_conv_layer(model_obj)
                            if last_conv:
                                heatmap = make_gradcam_heatmap(img_array, model_obj, last_conv)
                                g1, g2 = st.columns(2)
                                g1.image(image, caption="Original", use_container_width=True)
                                g2.image(
                                    overlay_heatmap(image, heatmap),
                                    caption=f"Grad-CAM ({last_conv})",
                                    use_container_width=True,
                                )
                            else:
                                st.warning("No convolutional layer was found for Grad-CAM.")
                except Exception as exc:
                    st.warning(f"Grad-CAM could not be generated: {exc}")

            # ---- AI Agronomist ----
            with tab_ai:
                st.session_state.assistant_disease_context = disease
                st.session_state.assistant_confidence_context = confidence

                if not st.session_state.gemini_api_connected:
                    st.write(
                        "Enter your Gemini API key to enable the assistant. The key is "
                        "verified, then this form is removed. It is kept in this session only."
                    )
                    with st.form("gemini_connect_form", clear_on_submit=False):
                        st.text_input(
                            "Gemini API key", type="password", key="gemini_api_key_entry",
                            help="Not written to a file or database.",
                        )
                        st.form_submit_button("Connect and verify", on_click=verify_gemini_api_key)
                    if st.session_state.gemini_api_status:
                        st.error(st.session_state.gemini_api_status)
                else:
                    with st.form("agronomist_request_form", clear_on_submit=True):
                        st.text_area(
                            "Your farming question",
                            placeholder=(
                                "How should I manage this disease?\n"
                                "What signs should I monitor?\n"
                                "What prevention steps should I take?"
                            ),
                            key="farmer_question_input",
                        )
                        st.form_submit_button("Ask Agronomist", on_click=submit_farmer_question)

                    if st.session_state.assistant_error:
                        st.error(st.session_state.assistant_error)
                    if st.session_state.assistant_answer:
                        with st.container(border=True):
                            st.markdown("#### Assistant response")
                            st.markdown(st.session_state.assistant_answer)

                    st.button(
                        "Disconnect API key", key="disconnect_gemini_api_button",
                        on_click=disconnect_gemini_api, type="secondary",
                    )

            st.caption(
                "For serious crop damage or uncertain results, consult a qualified "
                "agricultural officer or plant pathologist."
            )


# ============================================================
# PAGE: FIND AN EXPERT

# ============================================================

elif page == "Find an Expert":
    init_db()

    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_geocode(query):
        return geocode(query)

    def build_expert_map(df, center, zoom):
        kwargs = dict(
            lat="lat", lon="lon", color="Type", hover_name="Name",
            hover_data={"Specialization": True, "Organisation": True,
                        "lat": False, "lon": False, "Type": False},
            color_discrete_map={"Verified expert": T["chart"][0],
                                "Unverified expert": T["chart"][2],
                                "Your location": T["chart"][4]},
            center=center, zoom=zoom, height=480,
        )
        if hasattr(px, "scatter_map"):
            fig = px.scatter_map(df, **kwargs)
            fig.update_layout(map_style=T["map_style"])
        else:
            fig = px.scatter_mapbox(df, **kwargs)
            fig.update_layout(mapbox_style=T["map_style"])
        fig.update_traces(marker=dict(size=15))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.02, x=0, bgcolor=T["surface"],
                        font=dict(color=T["text"])),
        )
        return fig

    render_section_heading(
        "Expert directory",
        "Find an agriculture expert",
        "Browse registered agricultural officers, agronomists and plant disease "
        "specialists on the map, or register your own profile.",
    )

    def get_admin_password():
        try:
            secret = st.secrets.get("ADMIN_PASSWORD", "")
        except Exception:
            secret = ""
        return str(secret or os.getenv("ADMIN_PASSWORD") or DEFAULT_ADMIN_CODE)

    tabs = st.tabs(["Find experts", "Register as expert", "Admin"])

    # ---------------- FIND ----------------
    with tabs[0]:
        with st.container(border=True):
            f1, f2, f3 = st.columns([1.4, 1, 1])
            user_location = f1.text_input(
                "Your location (optional)",
                placeholder="Example: Seri Kembangan, Selangor, Malaysia",
                help="Used to sort experts by distance. It is only sent to OpenStreetMap to find coordinates.",
            ).strip()
            spec_filter = f2.selectbox("Specialization", ["Any"] + SPECIALIZATIONS)
            crop_filter = f3.selectbox("Crop", ["Any crop"] + CROPS)

        user_pos = cached_geocode(user_location) if user_location else None
        if user_location and user_pos is None:
            st.warning("Could not find that location. Try a nearby town or add the country.")

        experts = list_experts()
        if spec_filter != "Any":
            experts = [e for e in experts if e["specialization"] == spec_filter]
        if crop_filter != "Any crop":
            experts = [e for e in experts if crop_filter in (e["crops"] or "").split(",")]

        for e in experts:
            e["distance_km"] = (
                haversine_km(user_pos[0], user_pos[1], e["lat"], e["lon"]) if user_pos else None
            )
        if user_pos:
            experts.sort(key=lambda e: e["distance_km"])

        rows = [
            {
                "Name": e["name"],
                "Specialization": e["specialization"],
                "Organisation": e["organisation"] or "Independent",
                "lat": e["lat"], "lon": e["lon"],
                "Type": "Verified expert" if e["verified"] else "Unverified expert",
            }
            for e in experts
        ]
        if user_pos:
            rows.append({
                "Name": "You are here", "Specialization": "", "Organisation": user_location,
                "lat": user_pos[0], "lon": user_pos[1], "Type": "Your location",
            })

        if rows:
            map_df = pd.DataFrame(rows)
            if user_pos:
                center, zoom = {"lat": user_pos[0], "lon": user_pos[1]}, 9
            else:
                center = {"lat": map_df["lat"].mean(), "lon": map_df["lon"].mean()}
                zoom = 9 if len(map_df) == 1 else 4
        else:
            map_df = pd.DataFrame(
                [{"Name": "", "Specialization": "", "Organisation": "", "lat": 0.0,
                  "lon": 0.0, "Type": "Verified expert"}]
            ).iloc[0:0]
            center, zoom = {"lat": 10.0, "lon": 20.0}, 1.2

        st.plotly_chart(build_expert_map(map_df, center, zoom), use_container_width=True)

        st.markdown(f"### {len(experts)} expert{'s' if len(experts) != 1 else ''} found")
        if not experts:
            st.info("No registered experts match these filters yet. Use the **Register as expert** tab to add one.")

        for e in experts:
            with st.container(border=True):
                left, right = st.columns([3, 1.4])
                badge = "✅ Verified" if e["verified"] else "⚠️ Unverified"
                with left:
                    st.markdown(f"**{e['name']}** · {badge}")
                    st.caption(f"{e['specialization']} · {e['organisation'] or 'Independent'}")
                    st.write(f"📍 {e['location']}")
                    if e["crops"]:
                        st.write(f"🌱 {e['crops'].replace(',', ', ')}")
                    if e["notes"]:
                        st.write(e["notes"])
                with right:
                    if e["distance_km"] is not None:
                        st.metric("Distance", f"{e['distance_km']:.1f} km")
                    digits = re.sub(r"[^\d+]", "", e["phone"])
                    st.markdown(f"📞 [{e['phone']}](tel:{digits})")
                    if e["email"]:
                        st.markdown(f"✉️ [{e['email']}](mailto:{e['email']})")
                    st.markdown(
                        f"🗺️ [Open in Maps](https://www.google.com/maps/search/?api=1&query={e['lat']},{e['lon']})"
                    )

        st.caption(
            "Listings are submitted by experts themselves. Verify an expert's organisation "
            "and qualifications before sharing farm details or paying for a service."
        )

        with st.expander("Can't find an expert nearby? Search external directories"):
            if user_location:
                q = quote_plus(f"agriculture extension office {user_location}")
                st.link_button("Search Google Maps", f"https://www.google.com/maps/search/?api=1&query={q}")
                st.link_button(
                    "Find government agriculture offices",
                    "https://www.google.com/search?q=" + quote_plus(
                        "official government agriculture department contact " + user_location),
                )
            else:
                st.caption("Enter your location above to enable external search links.")

    # ---------------- REGISTER ----------------
    with tabs[1]:
        st.markdown("### Register as an agriculture expert")
        st.write(
            "Add your profile so farmers can find and contact you directly. "
            "New listings appear as **Unverified** until an administrator approves them."
        )

        with st.form("expert_register_form", clear_on_submit=True):
            r1, r2 = st.columns(2)
            reg_name = r1.text_input("Full name *")
            reg_spec = r2.selectbox("Specialization *", SPECIALIZATIONS)

            r3, r4 = st.columns(2)
            reg_org = r3.text_input("Organisation (optional)", placeholder="Department, company or university")
            reg_crops = r4.multiselect("Crops you advise on", CROPS, default=CROPS)

            r5, r6 = st.columns(2)
            reg_phone = r5.text_input("Phone number *", placeholder="+60 12 345 6789")
            reg_email = r6.text_input("Email (optional)")

            reg_location = st.text_input(
                "Work location *",
                placeholder="Office or town, e.g. Serdang, Selangor, Malaysia",
                help="Use your office or town. Avoid your home address.",
            )
            with st.expander("Set exact map coordinates (optional)"):
                st.caption("Only needed if the location above cannot be found automatically.")
                c1, c2 = st.columns(2)
                reg_lat = c1.number_input("Latitude", value=0.0, format="%.6f", min_value=-90.0, max_value=90.0)
                reg_lon = c2.number_input("Longitude", value=0.0, format="%.6f", min_value=-180.0, max_value=180.0)

            reg_notes = st.text_area("Short description (optional)", max_chars=300,
                                     placeholder="Services offered, languages spoken, working hours")
            reg_consent = st.checkbox(
                "I agree that my name, specialization, phone number, email and work location "
                "will be shown publicly in this directory. *"
            )
            reg_submit = st.form_submit_button("Register", use_container_width=True)

        if reg_submit:
            digits = re.sub(r"\D", "", reg_phone)
            if not reg_name.strip() or not reg_location.strip():
                st.error("Name and work location are required.")
            elif not 7 <= len(digits) <= 15:
                st.error("Enter a valid phone number (7 to 15 digits).")
            elif reg_email.strip() and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", reg_email.strip()):
                st.error("Enter a valid email address or leave it empty.")
            elif not reg_consent:
                st.error("Please accept the public listing agreement to register.")
            else:
                if reg_lat != 0.0 or reg_lon != 0.0:
                    coords = (reg_lat, reg_lon)
                else:
                    found = geocode(reg_location.strip())
                    coords = (found[0], found[1]) if found else None

                if coords is None:
                    st.error(
                        "The location could not be found on the map. Add the country, "
                        "or enter exact coordinates and submit again."
                    )
                else:
                    add_expert(
                        reg_name.strip(), reg_spec, reg_org.strip(), reg_crops,
                        reg_phone.strip(), reg_email.strip(), reg_location.strip(),
                        coords[0], coords[1], reg_notes.strip(),
                    )
                    st.success(
                        "Registration received. Your profile is now visible on the map "
                        "and marked Unverified until approved."
                    )

    # ---------------- ADMIN ----------------
    with tabs[2]:
        admin_password = get_admin_password()

        if not admin_password:
            st.info(
                "Admin access is not configured yet. Set a password to manage expert "
                "listings, then restart the app."
            )
            st.markdown("**Option 1: environment variable**")
            st.code("export ADMIN_PASSWORD='choose-a-strong-password'", language="bash")
            st.markdown("**Option 2: `.streamlit/secrets.toml`**")
            st.code('ADMIN_PASSWORD = "choose-a-strong-password"', language="toml")

        elif not st.session_state.get("expert_admin_ok"):
            attempts = st.session_state.get("expert_admin_attempts", 0)
            if attempts >= 5:
                st.error("Too many failed attempts. Reload the page to try again.")
            else:
                with st.form("expert_admin_login"):
                    pw = st.text_input("Admin password", type="password")
                    signed_in = st.form_submit_button("Sign in", use_container_width=True)
                if signed_in:
                    if hmac.compare_digest(pw, admin_password):
                        st.session_state.expert_admin_ok = True
                        st.session_state.expert_admin_attempts = 0
                        st.rerun()
                    else:
                        st.session_state.expert_admin_attempts = attempts + 1
                        st.error("Incorrect password.")

        else:
            head_l, head_r = st.columns([5, 1])
            head_l.markdown("### Expert administration")
            if head_r.button("Sign out", type="secondary", use_container_width=True):
                st.session_state.expert_admin_ok = False
                st.rerun()

            if admin_password == DEFAULT_ADMIN_CODE:
                st.warning(
                    "You are using the default admin code. Set the ADMIN_PASSWORD "
                    "environment variable (or secret) to a stronger password before "
                    "making the app public."
                )

            info = db_info()
            s1, s2, s3 = st.columns(3)
            s1.metric("Registered experts", info["total"])
            s2.metric("Verified", info["verified"])
            s3.metric("Pending review", info["total"] - info["verified"])

            admin_listings, admin_storage = st.tabs(["Listings", "Storage and backup"])

            with admin_listings:
                view = st.radio("Show", ["All", "Pending", "Verified"], horizontal=True)
                all_experts = list_experts()
                if view == "Pending":
                    all_experts = [e for e in all_experts if not e["verified"]]
                elif view == "Verified":
                    all_experts = [e for e in all_experts if e["verified"]]

                if not all_experts:
                    st.info("No listings to show.")

                for e in all_experts:
                    with st.container(border=True):
                        a1, a2, a3 = st.columns([4, 1.2, 1.2])
                        status = "✅ Verified" if e["verified"] else "⏳ Pending"
                        a1.markdown(
                            f"**{e['name']}** · {status}  \n"
                            f"{e['specialization']} · {e['organisation'] or 'Independent'}  \n"
                            f"{e['location']} · {e['phone']} · {e['email'] or 'no email'}  \n"
                            f"Registered {e['created_at']}"
                        )
                        if e["verified"]:
                            if a2.button("Unverify", key=f"unv_{e['id']}", use_container_width=True):
                                set_verified(e["id"], False)
                                st.rerun()
                        else:
                            if a2.button("Verify", key=f"ver_{e['id']}", use_container_width=True):
                                set_verified(e["id"], True)
                                st.rerun()

                        if st.session_state.get("expert_confirm_delete") == e["id"]:
                            if a3.button("Confirm", key=f"cdel_{e['id']}", use_container_width=True):
                                delete_expert(e["id"])
                                st.session_state.expert_confirm_delete = None
                                st.rerun()
                            if a3.button("Cancel", key=f"cancel_{e['id']}", type="secondary", use_container_width=True):
                                st.session_state.expert_confirm_delete = None
                                st.rerun()
                        else:
                            if a3.button("Delete", key=f"del_{e['id']}", type="secondary", use_container_width=True):
                                st.session_state.expert_confirm_delete = e["id"]
                                st.rerun()

            with admin_storage:
                st.markdown("#### Where data is stored")
                st.write(f"Database file: `{info['path']}`  ({info['size_kb']} KB)")
                st.caption(
                    "Change the location with the EXPERT_DB_PATH environment variable. "
                    "Free hosts such as Streamlit Community Cloud erase local files on "
                    "restart, so download a backup regularly or move to a hosted "
                    "database (Postgres or Supabase) for public use."
                )

                st.markdown("#### Backup")
                st.download_button(
                    "⬇️ Download backup (JSON)",
                    data=json.dumps(list_experts(), indent=2),
                    file_name=f"experts_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                )

                st.markdown("#### Restore")
                backup_file = st.file_uploader("Upload a backup file", type=["json"], key="expert_restore_file")
                if backup_file is not None and st.button("Restore from backup"):
                    try:
                        records = json.load(backup_file)
                        if not isinstance(records, list):
                            raise ValueError("The backup must contain a list of experts.")
                        added, skipped = import_experts(records)
                        st.success(f"Restored {added} expert(s). Skipped {skipped} duplicate or invalid entries.")
                    except Exception as exc:
                        st.error(f"Restore failed: {exc}")


# ============================================================
# PAGE: BUSINESS ANALYTICS (PROTOTYPE, SESSION-ONLY)
# ============================================================

elif page == "Business Analytics":
    render_section_heading(
        "Management overview",
        "Business Analytics",
        "Operational view of crop disease detections, model usage, and confidence.",
    )

    if st.session_state.history:
        analytics_df = pd.DataFrame(st.session_state.history).copy()
        analytics_df["Date and time"] = pd.to_datetime(analytics_df["Date and time"], errors="coerce")
        is_demo = False
    else:
        demo_rows = [
            ("2026-10-01", "Tomato___Early_blight", 92.4, "CNN", "Tomato"),
            ("2026-10-02", "Cassava___Mosaic", 88.1, "MobileNetV2", "Cassava"),
            ("2026-10-03", "Maize___Common_rust", 94.2, "CNN", "Maize"),
            ("2026-10-04", "Cashew___Leaf_spot", 86.7, "MobileNetV2", "Cashew"),
            ("2026-10-05", "Tomato___Late_blight", 91.3, "CNN", "Tomato"),
            ("2026-10-06", "Cassava___Brown_streak", 83.5, "MobileNetV2", "Cassava"),
            ("2026-10-07", "Maize___Northern_leaf_blight", 89.8, "CNN", "Maize"),
            ("2026-10-08", "Tomato___Healthy", 97.1, "CNN", "Tomato"),
            ("2026-10-09", "Cashew___Anthracnose", 87.6, "MobileNetV2", "Cashew"),
            ("2026-10-10", "Tomato___Early_blight", 90.2, "CNN", "Tomato"),
            ("2026-10-11", "Cassava___Mosaic", 85.9, "MobileNetV2", "Cassava"),
            ("2026-10-12", "Maize___Common_rust", 93.6, "CNN", "Maize"),
            ("2026-10-13", "Cashew___Leaf_spot", 82.8, "MobileNetV2", "Cashew"),
            ("2026-10-14", "Tomato___Late_blight", 89.4, "CNN", "Tomato"),
            ("2026-10-15", "Cassava___Healthy", 96.2, "MobileNetV2", "Cassava"),
            ("2026-10-16", "Maize___Northern_leaf_blight", 88.7, "CNN", "Maize"),
            ("2026-10-17", "Tomato___Early_blight", 91.8, "CNN", "Tomato"),
            ("2026-10-18", "Cashew___Anthracnose", 84.6, "MobileNetV2", "Cashew"),
            ("2026-10-19", "Maize___Common_rust", 92.1, "CNN", "Maize"),
            ("2026-10-20", "Tomato___Healthy", 98.0, "MobileNetV2", "Tomato"),
        ]
        analytics_df = pd.DataFrame(
            demo_rows, columns=["Date and time", "Disease", "Confidence (%)", "Model", "Crop"]
        )
        analytics_df["Date and time"] = pd.to_datetime(analytics_df["Date and time"])
        is_demo = True

    if "Crop" not in analytics_df.columns:
        def infer_crop(name):
            text = str(name).lower()
            return next((c for c in CROPS if c.lower() in text), "Other / Unspecified")
        analytics_df["Crop"] = analytics_df["Disease"].apply(infer_crop)

    if is_demo:
        st.info(
            "Preview: the figures below are illustrative sample data, not real "
            "records. Run predictions to see your current session data."
        )
    else:
        st.success("Showing prediction records from the current session only.")

    f1, f2, f3 = st.columns(3)
    with f1:
        opts = sorted(analytics_df["Crop"].astype(str).unique())
        sel_crops = st.multiselect("Crop", opts, default=opts)
    with f2:
        opts = sorted(analytics_df["Model"].astype(str).unique())
        sel_models = st.multiselect("Model", opts, default=opts)
    with f3:
        opts = sorted(analytics_df["Disease"].astype(str).unique())
        sel_diseases = st.multiselect("Disease class", opts, default=opts)

    filtered = analytics_df[
        analytics_df["Crop"].astype(str).isin(sel_crops)
        & analytics_df["Model"].astype(str).isin(sel_models)
        & analytics_df["Disease"].astype(str).isin(sel_diseases)
    ].copy()

    total = len(filtered)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total detections", total)
    k2.metric("Average confidence", f"{filtered['Confidence (%)'].mean():.1f}%" if total else "0.0%")
    k3.metric("Crop categories", int(filtered["Crop"].nunique()) if total else 0)
    k4.metric("Disease classes", int(filtered["Disease"].nunique()) if total else 0)

    st.markdown("### Detection distribution")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Detections by crop**")
        counts = filtered["Crop"].value_counts().rename_axis("Crop").reset_index(name="Detections")
        if counts.empty:
            st.caption("No records match the selected filters.")
        else:
            st.plotly_chart(styled_bar(counts, "Crop"), use_container_width=True)
    with ch2:
        st.markdown("**Most frequently detected disease classes**")
        counts = filtered["Disease"].value_counts().head(8).rename_axis("Disease").reset_index(name="Detections")
        if counts.empty:
            st.caption("No records match the selected filters.")
        else:
            st.plotly_chart(styled_bar(counts, "Disease", angle=-25), use_container_width=True)

    tr, md = st.columns(2)
    with tr:
        st.markdown("**Detection trend by day**")
        trend = filtered.dropna(subset=["Date and time"]).copy()
        if trend.empty:
            st.caption("Date information is not available for the selected records.")
        else:
            trend["Day"] = trend["Date and time"].dt.strftime("%Y-%m-%d")
            daily = trend.groupby("Day").size().reset_index(name="Detections")
            fig = px.line(
                daily, x="Day", y="Detections", markers=True,
                color_discrete_sequence=[T["chart"][0]], template=T["template"],
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8, color=T["chart"][2]))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=T["text"]), margin=dict(l=8, r=8, t=12, b=8),
                xaxis=dict(title=None, gridcolor=T["border"], linecolor=T["border"]),
                yaxis=dict(title="Detections", gridcolor=T["border"], zerolinecolor=T["border"]),
            )
            st.plotly_chart(fig, use_container_width=True)
    with md:
        st.markdown("**Model usage**")
        counts = filtered["Model"].value_counts().rename_axis("Model").reset_index(name="Detections")
        if counts.empty:
            st.caption("No model records match the selected filters.")
        else:
            st.plotly_chart(styled_bar(counts, "Model"), use_container_width=True)

    st.markdown("### Records for review")
    cols = ["Date and time", "Crop", "Disease", "Confidence (%)", "Model"]
    st.dataframe(
        filtered[cols].sort_values("Date and time", ascending=False),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download analytics as CSV",
        data=filtered[cols].to_csv(index=False).encode("utf-8"),
        file_name="crop_disease_analytics.csv",
        mime="text/csv",
    )
    st.caption(
        "Prototype limitation: this dashboard does not connect to a database, store "
        "records permanently, or calculate verified commercial KPIs such as revenue, "
        "yield, or farm-level loss."
    )


# ============================================================
# PAGE: PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":
    render_section_heading(
        "Activity log",
        "Prediction history",
        "Review the crop images analyzed during this session.",
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history).drop(columns=["Image hash"], errors="ignore")

        h1, h2 = st.columns(2)
        h1.metric("Total scans", len(history_df))
        h2.metric("Average confidence", f"{history_df['Confidence (%)'].mean():.2f}%")

        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear session history", type="secondary"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions recorded yet. Run a crop health scan to populate your history.")


# ============================================================
# PAGE: ABOUT
# ============================================================

else:
    render_section_heading(
        "Platform overview",
        "About",
        "A deep learning interface for preliminary crop leaf disease screening.",
    )

    st.markdown(
        """
        This system brings image-based crop disease screening and model
        explainability into one interface. Upload a leaf image or capture one
        with your camera, then run a prediction using the configured model.

        **Supported crops**
        - Cashew leaf
        - Cassava leaf
        - Maize leaf
        - Tomato leaf

        **Capabilities**
        - CNN and MobileNetV2 prediction
        - Side-by-side model comparison
        - Confidence and severity display
        - Disease descriptions, treatment, and prevention guidance
        - Grad-CAM explainability
        - PDF report generation
        - AI Agronomist question and answer
        - Session prediction history and CSV export

        **Important:** Predictions provide preliminary screening support only.
        Image quality, lighting, crop variety, and field conditions affect
        accuracy. Confirm important crop management decisions with an
        agricultural specialist.
        """
    )

    st.info(
        "To use the system, set the CNN and MobileNetV2 deployment folders in the "
        "sidebar. They must contain the files expected by your model_loader module."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(f'<div class="footer">{APP_NAME}</div>', unsafe_allow_html=True)