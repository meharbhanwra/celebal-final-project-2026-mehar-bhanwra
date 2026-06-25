from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image

from torchvision import models

from preprocessing import transform


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


def load_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(

        torch.load(

            CHECKPOINT_PATH,

            map_location=DEVICE

        )

    )

    model.to(DEVICE)

    model.eval()

    return model


def predict_image(

    image_file,

    model

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

        outputs = model(
            image
        )

        probabilities = torch.softmax(

            outputs,

            dim=1

        )

        probabilities = probabilities.squeeze().cpu()

        confidence, prediction = torch.max(
            probabilities,
            dim=0
        )

        top3_probabilities, top3_indices = torch.topk(
            probabilities,
            k=3
        )

        top3 = [

            (

                CLASS_NAMES[idx],

                prob.item()

            )

            for idx, prob in zip(

                top3_indices.tolist(),

                top3_probabilities

            )

        ]

        return (

            CLASS_NAMES[prediction.item()],

            confidence.item(),

            probabilities.numpy(),

            top3

        )