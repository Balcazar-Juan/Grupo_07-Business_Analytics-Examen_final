"""Cliente CoinGecko Demo. No imprime ni almacena la API key."""
from __future__ import annotations
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        key = os.getenv("COINGECKO_API_KEY", "").strip()
        if not key:
            raise RuntimeError("COINGECKO_API_KEY no configurada")
        self.session = requests.Session()
        self.session.headers.update({"accept": "application/json", "x-cg-demo-api-key": key})
        retries = Retry(total=2, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                        respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get(self, path, params=None):
        response = self.session.get(self.BASE_URL + path, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def historical_prices(self, coin_id, days=14):
        data = self._get(f"/coins/{coin_id}/market_chart", {
            "vs_currency": "usd", "days": min(max(int(days), 1), 365), "interval": "daily"
        })
        return data.get("prices", []) or []

    def find_id(self, coin_id, symbol, name):
        """Evita emparejar un token equivocado solo por coincidir su ticker."""
        coins = self._get("/search", {"query": name or symbol}).get("coins", [])
        for item in coins:
            if item.get("id", "").casefold() == str(coin_id).casefold():
                return item["id"]
        for item in coins:
            if (item.get("name", "").casefold() == str(name or "").casefold()
                    and item.get("symbol", "").casefold() == str(symbol or "").casefold()):
                return item["id"]
        return None
