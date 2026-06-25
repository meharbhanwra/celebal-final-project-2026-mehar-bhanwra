from PIL import Image
import pandas as pd

from torch.utils.data import Dataset


class EuroSATDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["filepath_rgb"]
        ).convert("RGB")

        label = int(
            row["class_id"]
        )

        if self.transform:
            image = self.transform(image)

        return image, label