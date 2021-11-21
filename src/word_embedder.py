import abc
from datetime import datetime

import fasttext
import gensim
import numpy as np
from transformers import AutoModel 


from src.utils import Singleton


class WordEmbedder(metaclass=Singleton):
    def __init__(self):
        self._model = None

    @abc.abstractmethod
    def __getitem__(self, word: str) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_dimension(self) -> int:
        pass


class FasttextWordEmbedder(WordEmbedder):
    def __init__(self, model_path):
        super().__init__()
        self._model_path = model_path

    def __getitem__(self, word: str) -> np.ndarray:
        if self._model is None:
            self._load_model()
        return self._model.get_word_vector(word)

    def get_dimension(self) -> int:
        if self._model is None:
            self._load_model()
        return self._model.get_dimension()

    def _load_model(self):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} FastText model loading started")
        self._model = fasttext.load_model(self._model_path)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} FastText model loading ended")


class Word2VecWordEmbedder(WordEmbedder):
    def __init__(self, model_path):
        super().__init__()
        self._model_path = model_path

    def __getitem__(self, word: str) -> np.ndarray:
        if self._model is None:
            self._load_model()

        if word in self._model.wv:
            return self._model.wv[word]
        else:
            return np.array([0] * self.get_dimension())

    def get_dimension(self) -> int:
        if self._model is None:
            self._load_model()
        return self._model.vector_size

    def _load_model(self):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Word2Vec model loading started")
        self._model = gensim.models.Word2Vec.load(self._model_path)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Word2Vec model loading ended")
        
        
class TransformersWordEmbedder(WordEmbedder):
    def __init__(self, model_path):
        super().__init__()
        self._model_path = model_path

    def __getitem__(self, words: str) -> np.ndarray:
        if self._model is None:
            self._load_model()
        return self._model(input_ids=words)
        
    def get_dimension(self) -> int:
        if self._model is None:
            self._load_model()
        return self._model.vector_size

    def _load_model(self):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Transformers model loading started")
        self._model = AutoModel.from_pretrained(self._model_path, output_hidden_states=True)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Transformers model loading ended")
