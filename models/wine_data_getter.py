from copy import deepcopy

import numpy as np
from dask.array import from_array
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from models.wine_dataset import WineDataSet


class WineMLDataGetter:
    def __init__(self, wine_dataset: WineDataSet, max_len, topn_varieties: int = 7, balance_class=False):
        filter_list = wine_dataset.varieties_count()['variety'][:topn_varieties].tolist()
        filtered_df = wine_dataset.data[wine_dataset.data['variety'].isin(filter_list)]

        if balance_class:
            aux_df = deepcopy(filtered_df)
            d = aux_df.groupby('variety')
            d = d.apply(lambda x: x.sample(d.size().min()).reset_index(drop=True))
            d = d.reset_index(drop=True)
            filtered_df = d
            del aux_df, d

        wine_embeddings_filter = filtered_df.index.values

        self._varieties_list = from_array(filter_list)
        self._wine_embeddings_filter = from_array(wine_embeddings_filter)

        self._variety2index = {variety: index for index, variety in enumerate(filter_list)}
        self._index2variety = {index: variety for index, variety in enumerate(filter_list)}
        # self._X = wine_embeddings[wine_embeddings_filter].compute()

        self._X = deepcopy(filtered_df['description_cleaned'].tolist())
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self._X)
        self._X, self._X_tokenizer = tokenizer.texts_to_sequences(self._X), tokenizer
        self._X = pad_sequences(self._X, maxlen=84, padding="pre", truncating="post")

        self._index2word = self._X_tokenizer.index_word
        self._word2index = self._X_tokenizer.word_index

        self._index2word.update({0: 'pad'})
        self._word2index.update({'pad': 0})

        self._Y = deepcopy(filtered_df['variety'])
        self._Y.replace(self._variety2index, inplace=True)
        self._Y = np.array(self._Y.tolist())

    @property
    def word2index(self):
        return self._word2index

    @property
    def index2word(self):
        return self._index2word

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y
