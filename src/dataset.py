import os
from typing import Tuple

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import get_label_encoder


class TextDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, avg_embedding: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.avg_embedding = avg_embedding

        self.train = None
        self.dev = None
        self.test = None
        self.setup()

    def setup(self):
        self.train = TextDataset(filepath=os.path.join(self.data_dir, 'train_set.csv'),
                                 avg_embedding=self.avg_embedding)
        self.dev = TextDataset(filepath=os.path.join(self.data_dir, 'dev_set.csv'),
                               avg_embedding=self.avg_embedding)
        self.test = TextDataset(filepath=os.path.join(self.data_dir, 'test_set.csv'),
                                avg_embedding=self.avg_embedding)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
        )


class TextDataset(Dataset):
    def __init__(self, filepath: str, avg_embedding: bool = False):
        super().__init__()
        df = pd.read_csv(filepath)

        self.text_data = df['text'].values
        if avg_embedding:
            self.embedding_data = df['embeddings'].values
        else:
            self.embedding_data = np.mean(df['embeddings'].values)

        self.label_encoder = get_label_encoder(df['labels'])
        self.labels = self.label_encoder.transform(df['labels'])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.embedding_data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
