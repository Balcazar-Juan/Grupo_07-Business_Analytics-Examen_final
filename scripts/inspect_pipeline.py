from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
from src.db import get_db, ping
from src.config import settings


def main():
    ping()
    db = get_db()

    active = list(db.assets.find(
        {"active_top_n": True},
        {"_id": 0, "coin_id": 1, "symbol": 1, "rank": 1, "history": 1, "is_stablecoin": 1}
    ).sort("rank", 1))

    print(f"BD: {settings.mongodb_db_name}")
    print(f"Activos Top N activos: {len(active)}")
    print(f"HISTORY_DAYS={settings.history_days} | FORECAST_DAYS={settings.forecast_days}")
    print()

    collections = [
        "assets", "market_snapshots", "mobula_snapshots", "price_history", "ohlcv",
        "binance_ohlcv", "market_daily", "news", "features_daily", "models",
        "model_metrics", "forecasts_daily", "forecast_runs", "data_quality", "pipeline_runs",
    ]
    print("=== CONTEO DE COLECCIONES ===")
    for name in collections:
        print(f"{name:22} {db[name].count_documents({}):>10}")

    print("\n=== COBERTURA POR ACTIVO ===")
    for a in active:
        h = a.get("history") or {}
        model = db.models.find_one({"coin_id": a["coin_id"]}, {"_id": 0, "model_version": 1})
        latest_forecast = db.forecast_runs.find_one(
            {"coin_id": a["coin_id"], "is_latest": True},
            {"_id": 0, "forecast_days": 1, "forecast_start": 1, "forecast_end": 1}
        )
        print(
            f"{str(a.get('rank') or '-'):>3} {str(a.get('symbol') or ''):<8} "
            f"rows={int(h.get('rows') or 0):>5} "
            f"coverage={float(h.get('coverage_pct') or 0):>6.2f}% "
            f"model={'YES' if model else 'NO ':<3} "
            f"forecast={int((latest_forecast or {}).get('forecast_days') or 0):>3}d"
        )

    print("\n=== FUENTES EN market_daily ===")
    pipeline = [
        {"$match": {"coin_id": {"$in": [a["coin_id"] for a in active]}}},
        {"$group": {"_id": "$source", "rows": {"$sum": 1}}},
        {"$sort": {"rows": -1}},
    ]
    for row in db.market_daily.aggregate(pipeline):
        print(f"{str(row.get('_id')):<30} {row.get('rows', 0):>10}")

    print("\n=== ESTADO ULTIMO FORECAST ===")
    statuses = Counter()
    for a in active:
        run = db.forecast_runs.find_one({"coin_id": a["coin_id"], "is_latest": True})
        statuses["forecasted" if run else "without_forecast"] += 1
    for k, v in statuses.items():
        print(f"{k:<20} {v}")


if __name__ == "__main__":
    main()
