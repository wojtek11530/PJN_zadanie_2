import os
from typing import List, Optional, Tuple

import nltk
import numpy as np
import spacy
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

from src.utils import get_label_encoder
from src.word_embedder import WordEmbedder, TransformersWordEmbedder

nlp = spacy.load("pl_core_news_md", disable=["parser", "ner"])


class TextDataModule(LightningDataModule):
    def __init__(self, data_dir: str, word_embedder: WordEmbedder,
                 batch_size: int = 64, avg_embedding: bool = False, preprocess_text: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.avg_embedding = avg_embedding
        self.word_embedder = word_embedder
        self.preprocess_text = preprocess_text

        self.train = None
        self.dev = None
        self.test = None
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.train = TextDataset(
            filepath=os.path.join(self.data_dir, 'hotels.sentence.train.txt'),
            word_embedder=self.word_embedder,
            avg_embedding=self.avg_embedding,
            preprocess_text=self.preprocess_text
        )
        self.dev = TextDataset(
            filepath=os.path.join(self.data_dir, 'hotels.sentence.dev.txt'),
            word_embedder=self.word_embedder,
            avg_embedding=self.avg_embedding,
            preprocess_text=self.preprocess_text
        )
        self.test = TextDataset(
            filepath=os.path.join(self.data_dir, 'hotels.sentence.test.txt'),
            word_embedder=self.word_embedder,
            avg_embedding=self.avg_embedding,
            preprocess_text=self.preprocess_text
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
    def __init__(self, filepath: str, word_embedder: WordEmbedder, avg_embedding: bool = False,
                 preprocess_text: bool = False):
        super().__init__()
        self.word_embedder = word_embedder
        self.preprocess_text = preprocess_text
        self.transformers_tokenizer = AutoTokenizer.from_pretrained("clarin-pl/roberta-polish-kgr10")

        texts, labels = self.get_texts_and_labels_from_file(self.read_txt(filepath))

        desc = f"Getting embeddings for {os.path.basename(filepath)}, preprocess={preprocess_text}"
        self.embedding_data = [self._get_embeddings_from_text(text) for text in tqdm(texts, desc=desc)]
        if avg_embedding:
            self.embedding_data = [np.mean(embeddings, axis=0) for embeddings in self.embedding_data]

        self.label_encoder = get_label_encoder(labels)
        self.labels = self.label_encoder.transform(labels).astype(np.int64)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.embedding_data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def _get_embeddings_from_text(self, text: str) -> np.ndarray:
        if type(self.word_embedder) != TransformersWordEmbedder:

            if self.preprocess_text:
                words = nlp(str(text))
                words = [
                    token.lemma_.lower() for token in words if not (
                            token.is_stop or token.is_punct or token.like_email or
                            token.like_url or token.like_num or token.is_digit or
                            token.pos_ not in ["NOUN", "ADJ", "VERB", "ADV"]
                    )
                ]
            else:
                words = nltk.word_tokenize(text)

            if len(words) > 0:
                embeddings = np.array([self.word_embedder[word] for word in words])
            else:
                # dodanie embeddingu złożonego z samych zer
                embeddings = np.array([[0] * self.word_embedder.get_dimension()])
            return embeddings
        else:
            words = self.transformers_tokenizer.encode(text)
            words = torch.LongTensor(words)
            words = words.unsqueeze(0)
            with torch.no_grad():
                out = self.word_embedder[words]
            hidden_states = out[2]
            sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
            return sentence_embedding.unsqueeze(0).detach().numpy()

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
