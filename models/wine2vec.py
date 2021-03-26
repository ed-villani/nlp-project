import logging
import os
from typing import Union, List

import numpy as np
from dask.array import from_array
from gensim.models import Word2Vec
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Wine2Vec:
    def __init__(self, sentences,
                 iter: int = 5,
                 min_count: int = 5,
                 size: int = 300,
                 workers: int = os.cpu_count() - 1,
                 sg: int = 1,
                 hs: int = 0,
                 negative: int = 5,
                 window: int = 5):
        self._sentences = sentences
        self._iter = iter
        self._min_count = min_count
        self._size = size
        self._workers = workers
        self._sg = sg
        self._hs = hs
        self._negative = negative
        self._window = window

        self._model = None
        self._wine_embedding = None

    @property
    def wine2vec(self):
        return self._wine_embedding

    @property
    def model(self) -> Word2Vec:
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def train(self, ):
        model = Word2Vec(
            sentences=self._sentences,
            iter=self._iter,
            min_count=self._min_count,
            size=self._size,
            workers=self._workers,
            sg=self._sg,
            hs=self._hs,
            negative=self._negative,
            window=self._window
        )

        self._model = model

    def save(self, path: str):
        self._model.save(f"{path}")

    def load(self, load_path: str):
        from gensim.models import Word2Vec
        self._model = Word2Vec.load(load_path)

    def similarity(self, w1, w2):
        return self._model.wv.similarity(w1, w2)

    def word(self, word):
        return self._model.wv[word]

    def most_similar(self, words: Union[str, List[str]], topn: int = 10):
        return self._model.wv.most_similar(words, topn=topn)

    def wine_embeddings(self):
        embeddings = []
        for sentence in tqdm(self._sentences):
            sentence_phrase = []
            for word in sentence:
                sentence_phrase.append(self.word(word))
            embeddings.append(np.array(sentence_phrase))
        wine_embedding = []
        for e in tqdm(embeddings):
            wine_embedding.append(np.mean(e, axis=0))
        wine_embedding = np.array(wine_embedding)
        self._wine_embedding = from_array(wine_embedding)
