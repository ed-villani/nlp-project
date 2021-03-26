from typing import Union

from models.wine import Wine


class WineDict:
    def __init__(self):
        self._data = {}
        self._title2index = {}
        self._index2title = {}
        self._repeated_wines_name = {}

    def append(self, wine: Wine):
        index = len(self._data)
        if wine.title in self._title2index:
            try:
                self._repeated_wines_name[wine.title].append(index)
            except KeyError:
                self._repeated_wines_name[wine.title] = [self._title2index[wine.title], index]
            wine.title = f"{wine.title} ({len(self._repeated_wines_name[wine.title])})"

        self._data[index] = wine
        self._index2title[index] = wine.title
        self._title2index[wine.title] = index

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            return self._data[key]
        return self._data[self._title2index[key]]

    def __str__(self):
        return str(self._data)

    __repr__ = __str__

    @property
    def title2index(self):
        return self._title2index

    @property
    def index2title(self):
        return self._index2title
