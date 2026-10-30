from __future__ import annotations
from datetime import datetime, timezone
from src.config import settings
from src.http_client import HttpClient

class BinanceClient:
    """Cliente publico de Binance Spot. No requiere API key para klines publicos."""

    def __init__(self):
        self.http = HttpClient()

    def klines(self, symbol: str, start_ms: int, end_ms: int, interval: str = "1d"):
        out = []
        cursor = int(start_ms)
        while cursor <= int(end_ms):
            r = self.http.request(
                "GET",
                f"{settings.binance_base_url}/api/v3/klines",
                params={
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": int(end_ms),
                    "limit": 1000,
                },
            )
            rows = r.json()
            if not isinstance(rows, list) or not rows:
                break
            out.extend(rows)
            last_open = int(rows[-1][0])
            next_cursor = last_open + 86_400_000
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(rows) < 1000:
                break
        return out

    @staticmethod
    def normalize_kline(row):
        return {
            "timestamp": datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "quote_volume": float(row[7]),
            "trades": int(row[8]),
        }
