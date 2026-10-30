from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
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
        if r.get("base_date") and pd.to_datetime(r["base_date"], utc=True) >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
    }
    run_ids = [r["forecast_run_id"] for r in runs.values()]
    forecasts = _frame(db.forecasts_daily.find({"forecast_run_id": {"$in": run_ids}}, {"_id": 0}), ["coin_id", "forecast_date", "predicted_price", "p10", "p90", "forecast_run_id"])
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"], utc=True)
    forecasts = forecasts.sort_values("forecast_date")
    return {"demo": False, "assets": assets, "history": history, "snapshots": snapshots, "quality": quality, "metrics": metrics, "runs": runs, "forecasts": forecasts}


def load_demo():
    """Serie reproducible solo para probar la interfaz; nunca representa mercado real."""
    rng = np.random.default_rng(7)
    assets = [
        {"coin_id": "bitcoin", "symbol": "BTC", "name": "Bitcoin", "rank": 1},
        {"coin_id": "ethereum", "symbol": "ETH", "name": "Ethereum", "rank": 2},
        {"coin_id": "binance-coin", "symbol": "BNB", "name": "BNB", "rank": 3},
        {"coin_id": "solana", "symbol": "SOL", "name": "Solana", "rank": 4},
    ]
    now = pd.Timestamp.now(tz="UTC").normalize()
    dates = pd.date_range(end=now, periods=400, freq="D")
    history, snapshots, forecasts, runs, quality = [], {}, [], {}, {}
    for asset, base, sigma in zip(assets, [75000, 2500, 650, 180], [0.025, 0.031, 0.027, 0.04]):
        cid = asset["coin_id"]
        prices = base * np.exp(np.cumsum(rng.normal(0.0003, sigma, len(dates))))
        for date, price in zip(dates, prices):
            history.append({"coin_id": cid, "date": date, "close": float(price), "volume": float(rng.uniform(1e8, 3e9)), "source": "DEMO"})
        snapshots[cid] = {"price": float(prices[-1]), "market_cap": float(prices[-1] * 1e7), "volume_24h": 1e9, "price_change_1d": float((prices[-1] / prices[-2] - 1) * 100), "observed_at": now.to_pydatetime()}
        runs[cid] = {"forecast_run_id": "demo-" + cid, "base_price": float(prices[-1]), "base_date": now.to_pydatetime(), "generated_at": now.to_pydatetime(), "forecast_days": 360}
        quality[cid] = {"actual_rows": 400, "requested_window_coverage_pct": 100.0, "status": "DEMO"}
        for horizon in range(1, 361):
            central = float(prices[-1] * np.exp(0.0003 * horizon))
            width = 1.28 * sigma * np.sqrt(horizon)
            forecasts.append({"coin_id": cid, "forecast_date": now + pd.Timedelta(days=horizon), "predicted_price": central, "p10": central * np.exp(-width), "p90": central * np.exp(width), "forecast_run_id": runs[cid]["forecast_run_id"]})
    return {"demo": True, "assets": assets, "history": pd.DataFrame(history), "snapshots": snapshots, "quality": quality, "metrics": {}, "runs": runs, "forecasts": pd.DataFrame(forecasts)}
