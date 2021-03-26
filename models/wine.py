import json


class Wine:
    def __init__(self,
                 points: int,
                 title: str,
                 description: dict,
                 taster_name: str,
                 taster_twitter_handle: str,
                 price: float,
                 designation: str,
                 variety: str,
                 region_1: str,
                 region_2: str,
                 province: str,
                 country: str,
                 winery: str,
                 ):
        self._data = {
            "points": int(points),
            "title": title,
            "description": description,
            "taster_name": taster_name,
            "taster_twitter_handle": taster_twitter_handle,
            "price": float(price),
            "designation": designation,
            "variety": variety,
            "region_1": region_1,
            "region_2": region_2,
            "province": province,
            "country": country,
            "winery": winery
        }

    def __str__(self):
        return json.dumps(self._data, indent=2)

    def __call__(self):
        return self._data

    __repr__ = __str__

    @property
    def title(self) -> str:
        return self._data['title']

    @title.setter
    def title(self, value):
        self._data['title'] = value

    @property
    def cleaned_description(self) -> str:
        return self._data['description']['cleaned']
