from __future__ import annotations
from datetime import datetime, timezone
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import ConfigurationError
from urllib.parse import urlsplit
import requests
from src.config import settings, require

_client = None

def _srv_fallback_uri(uri: str) -> str:
    """Resuelve SRV por HTTPS si el DNS local bloquea ese tipo de consulta."""
    parts = urlsplit(uri)
    if parts.scheme != "mongodb+srv":
        raise ValueError("El fallback solo admite mongodb+srv")
    response = requests.get(
        "https://dns.google/resolve",
        params={"name": "_mongodb._tcp." + parts.hostname, "type": "SRV"},
        timeout=8,
    )
    response.raise_for_status()
    hosts = [
        item["data"].split()[-1].rstrip(".")
        for item in response.json().get("Answer", [])
        if item.get("type") == 33
    ]
    if not hosts:
        raise ConfigurationError("No se encontraron hosts SRV de MongoDB Atlas")
    auth = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
    options = "tls=true&authSource=admin"
    if parts.query:
        options += "&" + parts.query
    return "mongodb://" + auth + ",".join(hosts) + (parts.path or "/") + "?" + options

def get_client():
    global _client
    require("mongodb_uri")
    if _client is None:
        try:
            _client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=15000, connectTimeoutMS=15000)
        except ConfigurationError:
            if not settings.mongodb_uri.startswith("mongodb+srv://"):
                raise
            _client = MongoClient(_srv_fallback_uri(settings.mongodb_uri), serverSelectionTimeoutMS=15000, connectTimeoutMS=15000)
    return _client

def get_db():
    return get_client()[settings.mongodb_db_name]

def ping():
    get_client().admin.command("ping")
    return True

def ensure_indexes():
    db = get_db()
    db.assets.create_index("coin_id", unique=True)
    db.assets.create_index([("rank", ASCENDING)])

    db.market_snapshots.create_index([("coin_id", ASCENDING), ("observed_at", DESCENDING)])
    db.mobula_snapshots.create_index([("coin_id", ASCENDING), ("observed_at", DESCENDING)])

    # RAW historico
    db.price_history.create_index(
        [("coin_id", ASCENDING), ("date", ASCENDING), ("source", ASCENDING)], unique=True
    )
    db.ohlcv.create_index(
        [("coin_id", ASCENDING), ("timestamp", ASCENDING), ("period", ASCENDING), ("source", ASCENDING)],
        unique=True,
    )
    db.binance_ohlcv.create_index(
        [("coin_id", ASCENDING), ("timestamp", ASCENDING), ("period", ASCENDING), ("source", ASCENDING)],
        unique=True,
    )

    # CURATED / Power BI
    db.market_daily.create_index([("coin_id", ASCENDING), ("date", ASCENDING)], unique=True)
    db.market_daily.create_index([("date", DESCENDING)])
    db.features_daily.create_index([("coin_id", ASCENDING), ("date", ASCENDING)], unique=True)
    db.data_quality.create_index([("coin_id", ASCENDING), ("checked_at", DESCENDING)])

    # Noticias
    db.news.create_index("news_id", unique=True)
    db.news.create_index([("published_at", DESCENDING)])
    db.news.create_index([("mentioned_coin_ids", ASCENDING), ("published_at", DESCENDING)])

    # ML
    db.models.create_index("coin_id", unique=True)
    db.model_metrics.create_index([("coin_id", ASCENDING), ("trained_at", DESCENDING)])
    db.forecast_runs.create_index("forecast_run_id", unique=True)
    db.forecasts_daily.create_index(
        [("forecast_run_id", ASCENDING), ("coin_id", ASCENDING), ("forecast_date", ASCENDING)], unique=True
    )
    db.forecasts_daily.create_index([("coin_id", ASCENDING), ("generated_at", DESCENDING), ("forecast_date", ASCENDING)])

    db.pipeline_runs.create_index([("started_at", DESCENDING)])
    db.ai_recommendations.create_index("evidence_hash", unique=True)
    db.ai_recommendations.create_index([("coin_id", ASCENDING), ("generated_at", DESCENDING)])

    # Capa relacional para Power BI / Atlas SQL
    db.dim_assets.create_index("coin_id", unique=True)
    db.data_quality_bi.create_index("coin_id", unique=True)
    db.news_bi.create_index("news_id", unique=True)
    db.bridge_news_assets.create_index([("news_id", ASCENDING), ("coin_id", ASCENDING)], unique=True)
    db.model_metrics_bi.create_index("coin_id", unique=True)
    db.model_feature_importance.create_index(
        [("coin_id", ASCENDING), ("model_version", ASCENDING), ("feature", ASCENDING)], unique=True
    )

def log_run(name: str, status: str, started_at, details=None, error=None):
    get_db().pipeline_runs.insert_one({
        "name": name,
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc),
        "details": details or {},
        "error": error,
    })
