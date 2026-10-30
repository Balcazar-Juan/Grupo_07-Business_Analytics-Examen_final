from src.config import settings, require
from src.http_client import HttpClient

class CoinStatsClient:
    def __init__(self):
        require("coinstats_api_key")
        self.http = HttpClient()
        self.headers = {"X-API-KEY": settings.coinstats_api_key, "Accept": "application/json"}

    def top_coins(self, limit=20):
        r = self.http.request(
            "GET",
            f"{settings.coinstats_base_url}/coins",
            headers=self.headers,
            params={
                "page": 1, "limit": limit, "currency": "USD",
                "sortBy": "rank", "sortDir": "asc"
            },
        )
        return r.json().get("result", [])
