from pathlib import Path

import pandas as pd

import streamlit as st

from inference import (
    load_model,
    predict_image
)

from feature_extractor import (
    load_feature_extractor,
    detect_change,
    generate_difference_heatmap
)

import torch

DEVICE = (

    "CUDA"

    if torch.cuda.is_available()

    else "CPU"

)

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
# Project Paths
# ==========================================================

APP_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = APP_DIR.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================================
# Constants
# ==========================================================

CLASS_NAMES = [

    "AnnualCrop",

    "Forest",

    "HerbaceousVegetation",

    "Highway",

    "Industrial",

    "Pasture",

    "PermanentCrop",

    "Residential",

    "River",

    "SeaLake"

]

CHANGE_THRESHOLD = 0.7274

# ==========================================================
# Cached Models
# ==========================================================

@st.cache_resource
def get_classifier():

    return load_model()


@st.cache_resource
def get_feature_extractor():

    return load_feature_extractor()


classifier = get_classifier()

feature_extractor = get_feature_extractor()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🛰️ Navigation")

page = st.sidebar.radio(

    "Choose a Module",

    (

        "🏷️ Land Use Classification",

        "🌍 Temporal Change Detection"

    )

)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")

st.sidebar.write("**Backbone:** ResNet18")

st.sidebar.write("**Dataset:** EuroSAT")

st.sidebar.write(
    f"**Change Threshold:** {CHANGE_THRESHOLD:.4f}"
)

st.sidebar.write(
    f"**Device:** {DEVICE}"
)

# ==========================================================
# Main Page
# ==========================================================

st.title("🛰️ Satellite Land Use Analysis")

st.markdown(
"""
This application demonstrates two computer vision tasks:

- **Land Use Classification**
- **Temporal Change Detection**
"""
)

st.divider()

# ==========================================================
# LAND USE CLASSIFICATION
# ==========================================================

if page == "🏷️ Land Use Classification":

    st.header("🏷️ Land Use Classification")

    st.markdown(
        "Upload a satellite image to predict its land-use class."
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:

        uploaded_file = st.file_uploader(

            "Upload Satellite Image",

            type=["jpg", "jpeg", "png", "tif"]

        )

        if uploaded_file is not None:

            st.image(

                uploaded_file,

                caption="Uploaded Image",

                use_container_width=True

            )

    with right_col:

        st.subheader("Prediction")

        if uploaded_file is None:

            st.info(
                "Prediction results will appear here."
            )

        else:

            predicted_class, confidence, probabilities, top3 = predict_image(
                uploaded_file,
                classifier
            )

            st.success(

                f"Predicted Class: **{predicted_class}**"

            )

            st.metric(

                "Confidence",

                f"{confidence:.2%}"

            )

            st.subheader("Top Predictions")

            for i, (cls, prob) in enumerate(top3, start=1):

                st.write(
                    f"**{i}. {cls}** — {prob:.2%}"
                )

            probability_df = pd.DataFrame({

                "Class": CLASS_NAMES,

                "Probability": probabilities

            })

            probability_df = probability_df.sort_values(

                "Probability",

                ascending=False

            ).set_index(

                "Class"

            )   

            st.subheader("Class Probabilities")

            st.bar_chart(

                probability_df

            )

# ==========================================================
# TEMPORAL CHANGE DETECTION
# ==========================================================

else:

    st.header("🌍 Temporal Change Detection")

    st.markdown(
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

        st.info(
            "Upload both images to begin change detection."
        )

    else:

        st.subheader("Uploaded Images")

        img1, img2 = st.columns(2)

        with img1:

            st.image(

                t1,

                caption="T1",

                use_container_width=True

            )

        with img2:

            st.image(

                t2,

                caption="T2",

                use_container_width=True

            )

        prediction, similarity = detect_change(

            t1,

            t2,

            feature_extractor,

            CHANGE_THRESHOLD

        )

        difference = generate_difference_heatmap(

            t1,

            t2

        )

        st.divider()

        st.subheader("Results")

        st.metric(

            "Cosine Similarity",

            f"{similarity:.4f}"

        )

        st.metric(

            "Threshold",

            f"{CHANGE_THRESHOLD:.4f}"

        )

        if prediction == "Changed":

            st.error("🔴 Prediction: Changed")

        else:

            st.success("🟢 Prediction: Unchanged")

        st.divider()

        st.subheader("Difference Heatmap")

        st.image(

            difference,

            caption="Absolute Pixel Difference",

            use_container_width=True

        )