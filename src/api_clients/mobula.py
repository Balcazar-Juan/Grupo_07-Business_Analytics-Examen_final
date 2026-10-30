from src.config import settings, require
from src.http_client import HttpClient

def batches(items, size=10):
    for i in range(0, len(items), size):
        yield items[i:i+size]

class MobulaClient:
    def __init__(self):
        require("mobula_api_key")
        self.http = HttpClient()
        self.headers = {
            "Authorization": settings.mobula_api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def search_asset(self, symbol, name=None):
        r = self.http.request(
            "GET",
            f"{settings.mobula_base_url}/2/fast-search",
            headers=self.headers,
            params={"input": f'"{symbol}"', "type": "assets", "sortBy": "marketCap", "limit": 10},
        )
        rows = r.json().get("data", []) or []
        exact = [x for x in rows if str(x.get("symbol", "")).upper() == str(symbol).upper()]
        if name:
            by_name = [x for x in exact if str(x.get("name", "")).lower() == str(name).lower()]
            if by_name:
                return by_name[0]
        # No asociar un activo de símbolo distinto: contaminaría años de historia.
        return exact[0] if exact else None

    def asset_details(self, asset_id):
        r = self.http.request(
            "GET", f"{settings.mobula_base_url}/2/asset/details",
            headers=self.headers, params={"id": int(asset_id), "tokensLimit": 10},
        )
        return r.json().get("data", {}) or {}

    def asset_details_batch(self, ids):
        out = []
        for batch in batches(ids):
            r = self.http.request(
                "POST", f"{settings.mobula_base_url}/2/asset/details",
                headers=self.headers,
                json=[{"id": int(x), "tokensLimit": 10} for x in batch],
            )
            payload = r.json()
            out.extend(payload.get("payload", payload.get("data", [])) or [])
        return out

    def price_history_batch(self, payload):
        out = []
        for batch in batches(payload):
            r = self.http.request(
                "POST", f"{settings.mobula_base_url}/2/asset/price-history",
                headers=self.headers, json=batch,
            )
            data = r.json().get("data", [])
            out.extend(data if isinstance(data, list) else [data])
        return out

    def token_ohlcv_batch(self, payload):
        out = []
        for batch in batches(payload):
            r = self.http.request(
                "POST", f"{settings.mobula_base_url}/2/token/ohlcv-history",
                headers=self.headers, json=batch,
            )
            data = r.json().get("data", [])
            out.extend(data if isinstance(data, list) else [data])
        return out
