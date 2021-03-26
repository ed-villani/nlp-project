from copy import deepcopy
from typing import Union

import pandas as pd
import numpy as np
from nltk import RegexpTokenizer


class WineDataSet:
    def __init__(self, filter_by_topn_varieties: Union[None, int] = None):
        self._df = pd.read_csv('inputs/wine_dataset/winemag-data-130k-v2.csv')
        if filter_by_topn_varieties is not None:
            self._df = self._df[
                self._df['variety'].isin(self._df.varieties_count()[:filter_by_topn_varieties]['variety'])]
        self._clean_description()

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        return np.array(self._df)

    @property
    def data(self):
        return deepcopy(self._df)

    @property
    def countries(self):
        return self._df['country'].unique()

    @property
    def varieties(self):
        return self._df['variety'].unique()

    def varieties_count(self):
        return self._df.groupby('variety').count()['country'] \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={'country': 'count'})

    def _clean_description(self):
        def remove_non_ascii(s):
            return "".join(i for i in s if ord(i) < 128)

        def make_lower_case(text):
            return text.lower()

        def remove_stop_words(text):
            from nltk.corpus import stopwords
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)
            return text

        def remove_punctuation(text):
            tokenizer = RegexpTokenizer(r'\w+')
            text = tokenizer.tokenize(text)
            text = " ".join(text)
            return text

        df = self._df
        df['description_cleaned'] = df['description'].apply(remove_non_ascii)
        df['description_cleaned'] = df.description_cleaned.apply(func=make_lower_case)
        df['description_cleaned'] = df.description_cleaned.apply(func=remove_stop_words)
        df['description_cleaned'] = df.description_cleaned.apply(func=remove_punctuation)
        df['description_cleaned'] = df['description_cleaned'].str.replace('\d+', '')
