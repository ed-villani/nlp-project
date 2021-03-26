import json
from typing import Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from models.row2json import Row2Json
from models.wine import Wine
from models.wine2vec import Wine2Vec
from models.wine_corpus import WineCorpus
from models.wine_dataset import WineDataSet
from models.wine_dict import WineDict


class WineRecommender:
    def __init__(self):
        self._wine130k = WineDataSet()
        self._wine_dict = WineDict()
        for row in tqdm(np.array(self._wine130k.data)):
            self._wine_dict.append(Wine(**Row2Json(row)))
        del row

        self._wine_corpus: Union[None, WineCorpus] = None
        self._wine2vec: Union[None, Wine2Vec] = None

    @property
    def wine2vec(self) -> Wine2Vec:
        return self._wine2vec

    @property
    def wine_corpus(self) -> WineCorpus:
        return self._wine_corpus

    @property
    def wine_dict(self) -> WineDict:
        return self._wine_dict

    @property
    def wine_dataset(self):
        return self._wine130k

    def build_corpus(self, save_path: Union[None, str] = None):
        self._wine_corpus = WineCorpus(self._wine_dict)
        if save_path is not None:
            self._wine_corpus.save(save_path)

    def load_corpus(self, load_path: str):
        self._wine_corpus = WineCorpus(load_path)

    def build_item2vec(self,
                       iter: int = 5,
                       min_count: int = 5,
                       size: int = 300,
                       window: int = 5,
                       save_path: Union[None, str] = None):

        self._wine2vec = Wine2Vec(
            sentences=self._wine_corpus.corpus,
            iter=iter,
            min_count=min_count,
            size=size,
            window=window
        )
        self._wine2vec.train()
        if save_path is not None:
            self._wine2vec.save(save_path)

    def load_item2vec(self, load_path: str):
        self._wine2vec = Wine2Vec(self._wine_corpus)
        self._wine2vec.load(load_path)
        if self._wine2vec.wine2vec is None:
            self._wine2vec.wine_embeddings()

    def build_gword2vec(self,
                        google_pretrained: str,
                        iter: int = 5,
                        min_count: int = 5,
                        size: int = 300,
                        window: int = 5,
                        save_path: Union[None, str] = None):
        # google_word2vec = KeyedVectors.load_word2vec_format(file_path, binary=True)
        from gensim.models import Word2Vec
        import os
        self._wine2vec = Wine2Vec(
            sentences=self._wine_corpus.corpus,
            min_count=min_count,
            size=size,
            window=window
        )

        self._wine2vec.model = Word2Vec(
            min_count=min_count,
            size=size,
            workers=os.cpu_count() - 1,
            window=window
        )

        self._wine2vec.model.build_vocab(self._wine_corpus)
        self._wine2vec.model.intersect_word2vec_format(google_pretrained,
                                                       lockf=1.0, binary=True)
        self._wine2vec.model.train(self._wine_corpus.corpus, total_examples=self._wine2vec.model.corpus_count,
                                   epochs=iter)
        if save_path is not None:
            self._wine2vec.save(save_path)

    def recommend(self, wine: int):
        import json
        wine_index = None
        if isinstance(wine, str):
            wine_index = self._wine_dict.title2index[wine]
        elif isinstance(wine, int):
            wine_index = wine
        if self._wine2vec.wine2vec is None:
            self._wine2vec.wine_embeddings()
        wine_embeddings = self._wine2vec.wine2vec[wine_index:wine_index + 1]
        similarities = cosine_similarity(wine_embeddings, self._wine2vec.wine2vec)
        bigger_similarity = np.argsort(-similarities)[0][1]
        return json.dumps(self._wine_dict[int(bigger_similarity)](), indent=2)

    def recommend_by_description(self, description: str):
        words = description.split()
        words_embeddings = []
        for w in words:
            try:
                words_embeddings.append(self._wine2vec.model[w])
            except KeyError:
                pass

        words_embeddings = np.array(words_embeddings)
        similarities = cosine_similarity(words_embeddings, self._wine2vec.wine2vec)
        bigger_similarity = np.argsort(-similarities)[0][0]
        return json.dumps(self._wine_dict[int(bigger_similarity)](), indent=2)