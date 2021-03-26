class Row2Json:
    def __new__(cls, row):
        return {
            "points": int(row[4]),
            "title": row[11] if not isinstance(row[11], float) else None,
            "description": {
                'original': row[2] if not isinstance(row[2], float) else None,
                'cleaned': row[14] if not isinstance(row[14], float) else None
            },
            "taster_name": row[9] if not isinstance(row[9], float) else None,
            "taster_twitter_handle": row[10] if not isinstance(row[10], float) else None,
            "price": float(row[5]),
            "designation": row[3] if not isinstance(row[3], float) else None,
            "variety": row[12] if not isinstance(row[12], float) else None,
            "region_1": row[7] if not isinstance(row[7], float) else None,
            "region_2": row[8] if not isinstance(row[8], float) else None,
            "province": row[6] if not isinstance(row[9], float) else None,
            "country": row[1] if not isinstance(row[1], float) else None,
            "winery": row[13] if not isinstance(row[13], float) else None
        }
