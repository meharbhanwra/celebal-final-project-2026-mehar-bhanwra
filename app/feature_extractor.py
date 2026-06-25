from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity

from torchvision import models

from preprocessing import transform

import cv2


DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (

    PROJECT_ROOT

    / "checkpoints"

    / "resnet18_finetuned_best.pt"

)


def load_feature_extractor():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    model.load_state_dict(

        torch.load(

            CHECKPOINT_PATH,

            map_location=DEVICE

        )

    )

    feature_extractor = nn.Sequential(

        *list(model.children())[:-1]

    )

    feature_extractor.to(
        DEVICE
    )

    feature_extractor.eval()

    return feature_extractor


def extract_embedding(

    image_file,

    feature_extractor

):

    image = Image.open(
        image_file
    ).convert(
        "RGB"
    )

    image = transform(
        image
    )

    image = image.unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.no_grad():

        embedding = feature_extractor(
            image
        )

    return (

        embedding

        .flatten()

        .cpu()

        .numpy()

    )


def compute_similarity(

    image1,

    image2,

    feature_extractor

):

    emb1 = extract_embedding(

        image1,

        feature_extractor

    )

    emb2 = extract_embedding(

        image2,

        feature_extractor

    )

    similarity = cosine_similarity(

        emb1.reshape(1, -1),

        emb2.reshape(1, -1)

    )[0][0]

    return similarity


def detect_change(

    image1,

    image2,

    feature_extractor,

    threshold=0.7274

):

    similarity = compute_similarity(

        image1,

        image2,

        feature_extractor

    )

    prediction = (

        "Changed"

        if similarity < threshold

        else "Unchanged"

    )

    return prediction, similarity

def generate_difference_heatmap(
    image1,
    image2
):

    img1 = Image.open(
        image1
    ).convert(
        "RGB"
    ).resize(
        (224, 224)
    )

    img2 = Image.open(
        image2
    ).convert(
        "RGB"
    ).resize(
        (224, 224)
    )

    img1 = np.array(img1)
    img2 = np.array(img2)

    diff = np.abs(

        img1.astype(np.int16)

        -

        img2.astype(np.int16)

    ).astype(np.uint8)

    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_RGB2GRAY
    )

    heatmap = cv2.applyColorMap(

        gray,

        cv2.COLORMAP_JET

    )

    heatmap = cv2.cvtColor(

        heatmap,

        cv2.COLOR_BGR2RGB

    )

    return heatmap