from typing import Union

import numpy as np
from tqdm import tqdm

from models.wine_dict import WineDict


class WineCorpus:
    def __init__(self, text: Union[str, WineDict]):
        corpus = []
        sentence_len = []

        self._corpus = None
        self._max_len = None
        self._min_len = None
        self._avg_len = None
        self._std_len = None

        if isinstance(text, WineDict):
            for w in tqdm(text):
                wine = text[w]
                cd = wine.cleaned_description.split()
                sentence_len.append(len(cd))
                corpus.append(cd)
            self._corpus = corpus
            del text, w
        elif isinstance(text, str):
            with open(text, 'r') as f:
                for sentence in tqdm(f.readlines()):
                    corpus.append(sentence.replace('\n', '').split())
                    sentence_len.append(len(sentence.split()))
                self._corpus = corpus
            f.close()
        self.init_data(sentence_len)

    def init_data(self, sentence_len):
        sentence_len = np.array(sentence_len)
        self._max_len = max(sentence_len)
        self._min_len = min(sentence_len)
        self._avg_len = int(np.mean(sentence_len))
        self._std_len = int(np.std(sentence_len))

    def __iter__(self):
        return self._corpus.__iter__()

    @property
    def corpus(self):
        return self._corpus

    @property
    def max_len(self):
        return self._max_len

    @property
    def min_len(self):
        return self._min_len

    @property
    def avg_len(self):
        return self._avg_len

    @property
    def std_len(self):
        return self._std_len

    def save(self, save_path: str):
        with open(save_path, "w") as f:
            for sentence in self._corpus:
                s = " ".join(map(str, sentence))
                f.write(f"{s}\n")
        f.close()

