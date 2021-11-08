import os
from typing import List, Optional, Tuple

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import get_label_encoder
from src.word_embedder import WordEmbedder


class TextDataModule(LightningDataModule):
    def __init__(self, data_dir: str, word_embedder: WordEmbedder,
                 batch_size: int = 64, avg_embedding: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.avg_embedding = avg_embedding
        self.word_embedder = word_embedder

        self.train = None
        self.dev = None
        self.test = None
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.train = TextDataset(
            filepath=os.path.join(self.data_dir, 'hotels.sentence.train.pl.txt'),
            word_embedder=self.word_embedder,
            avg_embedding=self.avg_embedding
        )
        self.dev = TextDataset(
            filepath=os.path.join(self.data_dir, 'hotels.sentence.dev.pl.txt'),
            word_embedder=self.word_embedder,
            avg_embedding=self.avg_embedding
        )
        self.test = TextDataset(
            filepath=os.path.join(self.data_dir, 'hotels.sentence.test.pl.txt'),
            word_embedder=self.word_embedder,
            avg_embedding=self.avg_embedding
        )

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
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
        )


class TextDataset(Dataset):
    def __init__(self, filepath: str, word_embedder: WordEmbedder, avg_embedding: bool = False):
        super().__init__()
        self.word_embedder = word_embedder

        texts, labels = self.get_texts_and_labels_from_file(self.read_txt(filepath))

        self.embedding_data = [self._get_embeddings_from_text(text) for text in texts]
        if avg_embedding:
            self.embedding_data = [np.mean(embeddings, axis=0) for embeddings in self.embedding_data]

        self.label_encoder = get_label_encoder(labels)
        self.labels = self.label_encoder.transform(labels).astype(np.int64)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.embedding_data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def _get_embeddings_from_text(self, text: str) -> np.ndarray:
        words = text.split(' ')
        embeddings = np.array([self.word_embedder[word] for word in words])
        return embeddings

    @staticmethod
    def read_txt(input_file: str) -> List[str]:
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='UTF-8') as f:
            lines = f.read().splitlines()
        return lines

    @staticmethod
    def get_texts_and_labels_from_file(lines) -> Tuple[np.ndarray, np.ndarray]:
        texts = []
        labels = []
        for (i, line) in enumerate(lines):
            split_line = line.split('__label__')
            text = split_line[0]
            label = split_line[1]
            texts.append(text)
            labels.append(label)

        return np.array(texts), np.array(labels)
