from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime, timezone
from src.db import ensure_indexes, log_run
from src.etl.market import refresh_market
from src.etl.news import refresh_news
from src.etl.history import load_price_history, load_ohlcv, load_binance_ohlcv, build_market_daily
from src.ml.model import train_all, forecast_all
from src.config import settings
from src.etl.powerbi import publish_powerbi_layer


def main(daily=False, train=False):
    started = datetime.now(timezone.utc)
    details = {"daily": daily, "train": train}
    run_name = "refresh_model" if train else ("refresh_daily" if daily else "refresh")
    try:
        ensure_indexes()
        details["market"] = refresh_market()
        details["news"] = refresh_news()

        if daily or train:
            days = settings.incremental_history_days
            details["history_days_requested"] = days
            details["price_history"] = load_price_history(days=days)
            try:
                details["mobula_ohlcv"] = load_ohlcv(days=days)
            except Exception as exc:
                details["mobula_ohlcv"] = {"warning": str(exc)}
            try:
                details["binance_ohlcv"] = load_binance_ohlcv(days=days)
            except Exception as exc:
                details["binance_ohlcv"] = {"warning": str(exc)}
            # Reconstruye la capa curada usando todo el horizonte almacenado.
            details["market_daily"] = build_market_daily(days=settings.history_days)

        if train:
            trained = train_all()
            details["models_trained"] = sum(x.get("status") == "trained" for x in trained)
            forecasts = forecast_all()
            details["forecasts"] = sum(x.get("status") == "forecasted" for x in forecasts)
            details["forecast_days"] = settings.forecast_days

        details["powerbi"] = publish_powerbi_layer()

        log_run(run_name, "success", started, details)
        print("OK", details)
    except Exception as exc:
        log_run(run_name, "error", started, details, str(exc))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--daily", action="store_true", help="Actualiza historicos incrementales y market_daily")
    p.add_argument("--train", action="store_true", help="Ademas reentrena XGBoost y genera el forecast")
    args = p.parse_args()
    main(daily=args.daily or args.train, train=args.train)
