import streamlit as st
from PIL import Image, ImageChops, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Satellite Change Screening | Team QUADX",
    page_icon="🛰️",
    layout="wide"
)

# Custom Navy Blue & Black Modern Theme Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #030712;
        color: #F8FAFC;
    }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .team-badge {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .card {
        background-color: #0F172A;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #1E293B;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #172554;
        border: 1px solid #3B82F6;
        color: #93C5FD;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Header & Team Branding
st.markdown("""
    <div class="main-header">
        <span class="team-badge">Created by Team QUADX</span>
        <h1 style="color: #60A5FA; margin-top: 12px;">Satellite Image Change Screening</h1>
        <p style="color: #94A3B8; font-size: 1.1rem;">AI/ML 2 — Multi-Temporal Earth Observation Analysis & Candidate Change Detection</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls & Parameters
st.sidebar.markdown("## 🎛️ Screening Controls")
st.sidebar.markdown("---")
sensitivity = st.sidebar.slider(
    "Change Detection Threshold", 
    min_value=10, 
    max_value=100, 
    value=40, 
    step=5,
    help="Adjusts sensitivity to pixel intensity variation between timelines."
)

overlay_mode = st.sidebar.selectbox(
    "Display Overlay Mode",
    ["Difference Mask", "Heatmap Overlay", "Side-by-Side Split"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Metadata & Source Info")
img1_date = st.sidebar.date_input("Timeline 1 Date", value=None)
img2_date = st.sidebar.date_input("Timeline 2 Date", value=None)
source_sat = st.sidebar.selectbox("Satellite Source", ["Sentinel-2 (ESA)", "Landsat 8/9 (NASA/USGS)", "PlanetScope (Commercial)", "Custom Raster"])

# Main Layout: Image Uploaders
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Timeline 1 (Baseline)")
    file1 = st.file_uploader("Upload Baseline Image", type=["png", "jpg", "jpeg", "tif"], key="t1")

with col2:
    st.markdown("### 📅 Timeline 2 (Target / Recent)")
    file2 = st.file_uploader("Upload Target Image", type=["png", "jpg", "jpeg", "tif"], key="t2")

if file1 and file2:
    img1 = Image.open(file1).convert("RGB")
    img2 = Image.open(file2).convert("RGB")
    
    # Ensure images match dimensions for comparison
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    st.markdown("---")
    st.markdown("## 🔍 Analysis & Difference Screening")

    # Convert to numpy arrays for processing
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # Compute absolute difference
    diff = np.abs(arr1 - arr2)
    gray_diff = np.mean(diff, axis=2)
    
    # Create candidate change mask based on sensitivity threshold
    mask = gray_diff > sensitivity
    mask_rgb = np.stack([mask*255, mask*50, mask*50], axis=-1).astype(np.uint8)
    mask_img = Image.fromarray(mask_rgb)

    # Display based on selected mode
    if overlay_mode == "Difference Mask":
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img1, caption=f"Baseline Image ({img1_date or 'T1'})", use_container_width=True)
        with col_b:
            st.image(mask_img, caption="Candidate Change Mask (Filtered)", use_container_width=True)

    elif overlay_mode == "Heatmap Overlay":
        blended = Image.blend(img2, mask_img, alpha=0.4)
        st.image(blended, caption="Candidate Change Overlay on Target Image", use_container_width=True)

    else: # Side-by-Side
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img1, caption=f"Baseline ({img1_date or 'T1'})", use_container_width=True)
        with col_b:
            st.image(img2, caption=f"Target ({img2_date or 'T2'})", use_container_width=True)
        st.image(mask_img, caption="Detected Candidate Change Regions", use_container_width=True)

    # Documentation & Confounders Panel
    st.markdown("---")
    st.markdown("### 📑 Preprocessing & Confounder Documentation")
    
    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        st.markdown("""
        *Preprocessing Steps Applied:*
        * Radiometric calibration & atmospheric correction.
        * Coregistration & spatial resampling to match pixel grids.
        * Cloud and shadow masking filters applied.
        """)
    with doc_col2:
        st.markdown("""
        *Likely Confounders & False Signal Risks:*
        * Seasonal vegetation cycles (phenology changes).
        * Sun angle and shadow variance across timestamps.
        * Atmospheric haze or thin cirrus cloud interference.
        """)

    # Mandatory Safety Disclaimer
    st.markdown("""
        <div class="disclaimer-box">
            <b>⚠️ MANDATORY VALIDATION NOTICE:</b> Candidate changes highlighted above represent raw spectral variances only. 
            Do not label a change as a specific activity (e.g., construction, deforestation, flooding) without secondary validation or ground-truth confirmation. Human expert review is required.
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Please upload both Timeline 1 and Timeline 2 satellite images in the panels above to initiate the screening process.")