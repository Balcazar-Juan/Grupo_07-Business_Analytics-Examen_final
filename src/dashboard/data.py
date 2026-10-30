from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd

from src.db import get_db


def _frame(rows, columns):
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    return frame


def load_real():
    """Lee solo colecciones curadas y el último pronóstico por moneda."""
    db = get_db()
    db.client.admin.command("ping")
    assets = list(db.assets.find({"active_top_n": True}, {"_id": 0, "coin_id": 1, "symbol": 1, "name": 1, "rank": 1, "is_stablecoin": 1}).sort("rank", 1))
    if not assets:
        raise ValueError("MongoDB está conectado, pero assets aún no tiene monedas. Ejecuta scripts/bootstrap.py.")
    ids = [a["coin_id"] for a in assets]
    since = datetime.now(timezone.utc) - timedelta(days=400)
    history = _frame(db.market_daily.find({"coin_id": {"$in": ids}, "date": {"$gte": since}}, {"_id": 0}), ["coin_id", "date", "close", "volume", "source"])
    history["date"] = pd.to_datetime(history["date"], utc=True)
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history = history[(history["close"] > 0) & history["date"].notna()].sort_values("date").drop_duplicates(["coin_id", "date"], keep="last")

    snapshots = {}
    for row in db.market_snapshots.find({"coin_id": {"$in": ids}}, {"_id": 0}).sort("observed_at", -1):
        snapshots.setdefault(row["coin_id"], row)
        if len(snapshots) == len(ids):
            break
    quality = {}
    for row in db.data_quality.find({"coin_id": {"$in": ids}}, {"_id": 0}).sort("checked_at", -1):
        quality.setdefault(row["coin_id"], row)
        if len(quality) == len(ids):
            break
    metrics = {}
    for row in db.model_metrics.find({"coin_id": {"$in": ids}}, {"_id": 0, "folds": 0, "feature_importance": 0}).sort("trained_at", -1):
        metrics.setdefault(row["coin_id"], row)
        if len(metrics) == len(ids):
            break
    runs = {
        r["coin_id"]: r
        for r in db.forecast_runs.find({"coin_id": {"$in": ids}, "is_latest": True}, {"_id": 0})
        if r.get("base_date") and pd.to_datetime(r["base_date"], utc=True) >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    }
    run_ids = [r["forecast_run_id"] for r in runs.values()]
    forecasts = _frame(db.forecasts_daily.find({"forecast_run_id": {"$in": run_ids}}, {"_id": 0}), ["coin_id", "forecast_date", "predicted_price", "p10", "p90", "forecast_run_id"])
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"], utc=True)
    forecasts = forecasts.sort_values("forecast_date")
    news = _frame(db.news.find(
        {"published_at": {"$gte": datetime.now(timezone.utc) - timedelta(days=30)}},
        {"_id": 0, "title": 1, "url": 1, "source": 1, "published_at": 1,
         "mentioned_coin_ids": 1, "sentiment_compound": 1, "sentiment_label": 1}
    ).sort("published_at", -1).limit(300), ["title", "url", "source", "published_at", "mentioned_coin_ids", "sentiment_compound", "sentiment_label"])
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
    news["sentiment_compound"] = pd.to_numeric(news["sentiment_compound"], errors="coerce")
    recommendations = {}
    # Include older saved verdicts that predate the is_latest marker. Newest wins.
    for row in db.ai_recommendations.find({"coin_id": {"$in": ids}}, {"_id": 0}).sort("generated_at", -1):
        recommendations.setdefault(row["coin_id"], row)
    return {"assets": assets, "history": history, "snapshots": snapshots, "quality": quality, "metrics": metrics, "runs": runs, "forecasts": forecasts, "news": news, "recommendations": recommendations}
