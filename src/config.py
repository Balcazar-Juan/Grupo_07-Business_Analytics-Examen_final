from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default

def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default

@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "crypto_analytics")
    coinstats_api_key: str = os.getenv("COINSTATS_API_KEY", "")
    mobula_api_key: str = os.getenv("MOBULA_API_KEY", "")

    top_n: int = env_int("TOP_N", 20)
    # 5 anios aprox. Se puede cambiar sin tocar codigo.
    history_days: int = env_int("HISTORY_DAYS", 1826)
    # Horizonte futuro configurable. Por ahora: 360 dias.
    forecast_days: int = env_int("FORECAST_DAYS", 365)
    model_min_rows: int = env_int("MODEL_MIN_ROWS", 365)
    forecast_return_clip: float = env_float("FORECAST_RETURN_CLIP", 0.25)

    # Recarga incremental diaria: evita pedir nuevamente los 5 anios.
    incremental_history_days: int = env_int("INCREMENTAL_HISTORY_DAYS", 14)

    coinstats_base_url: str = "https://openapiv1.coinstats.app"
    mobula_base_url: str = "https://api.mobula.io/api"
    binance_base_url: str = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")

    rss_coindesk: str = os.getenv("RSS_COINDESK", "https://www.coindesk.com/arc/outboundfeeds/rss/")
    rss_cointelegraph: str = os.getenv("RSS_COINTELEGRAPH", "https://cointelegraph.com/rss")
    rss_decrypt: str = os.getenv("RSS_DECRYPT", "https://decrypt.co/feed")

settings = Settings()

def require(*attrs: str) -> None:
    names = {
        "mongodb_uri": "MONGODB_URI",
        "coinstats_api_key": "COINSTATS_API_KEY",
        "mobula_api_key": "MOBULA_API_KEY",
    }
    missing = [names.get(a, a) for a in attrs if not getattr(settings, a)]
    if missing:
        raise RuntimeError("Faltan variables de entorno: " + ", ".join(missing))
