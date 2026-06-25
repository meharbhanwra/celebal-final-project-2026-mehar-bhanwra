from pathlib import Path

import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Satellite Land Use Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Paths
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🛰️ Navigation")

page = st.sidebar.radio(
    "Choose a Module",
    [
        "🏷️ Land Use Classification",
        "🌍 Temporal Change Detection"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:** ResNet18")
st.sidebar.write("**Dataset:** EuroSAT")
st.sidebar.write("**Change Threshold:** 0.7274")

# ==========================================================
# Main Page
# ==========================================================

st.title("🛰️ Satellite Land Use Analysis")

st.markdown("""
This application demonstrates two computer vision tasks:

- **Land Use Classification**
- **Temporal Change Detection**
""")

st.divider()

# ==========================================================
# Classification Page
# ==========================================================

if page == "🏷️ Land Use Classification":

    st.header("🏷️ Land Use Classification")

    st.write(
        "Upload a satellite image to predict its land-use class."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "tif"]
    )

    if uploaded_file is None:
        st.info("Please upload an image to continue.")

# ==========================================================
# Change Detection Page
# ==========================================================

else:

    st.header("🌍 Temporal Change Detection")

    st.write(
        "Upload two satellite images captured at different times."
    )

    col1, col2 = st.columns(2)

    with col1:
        t1 = st.file_uploader(
            "T1 Image",
            type=["jpg", "jpeg", "png", "tif"],
            key="t1"
        )

    with col2:
        t2 = st.file_uploader(
            "T2 Image",
            type=["jpg", "jpeg", "png", "tif"],
            key="t2"
        )

    if t1 is None or t2 is None:
        st.info("Upload both images to continue.")