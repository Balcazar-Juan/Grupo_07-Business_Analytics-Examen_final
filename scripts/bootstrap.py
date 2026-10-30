from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
from src.db import ensure_indexes, log_run
from src.etl.market import refresh_market
from src.etl.history import load_price_history, load_ohlcv, load_binance_ohlcv, build_market_daily
from src.etl.news import refresh_news
from src.ml.model import train_all, forecast_all
from src.config import settings
from src.etl.powerbi import publish_powerbi_layer

if __name__ == "__main__":
    started = datetime.now(timezone.utc)
    details = {"history_days": settings.history_days, "forecast_days": settings.forecast_days}
    try:
        ensure_indexes()

        print("1/8 Mercado y maestro de activos")
        details["market"] = refresh_market()
        print(details["market"])

        print(f"2/8 Historico RAW Mobula ({settings.history_days} dias)")
        details["price_history"] = load_price_history()
        print(details["price_history"])

        print("3/8 OHLCV RAW Mobula")
        try:
            details["mobula_ohlcv"] = load_ohlcv()
        except Exception as exc:
            details["mobula_ohlcv"] = {"warning": str(exc)}
        print(details["mobula_ohlcv"])

        print("4/8 OHLCV RAW Binance (fallback publico)")
        try:
            details["binance_ohlcv"] = load_binance_ohlcv()
        except Exception as exc:
            details["binance_ohlcv"] = {"warning": str(exc)}
        print(details["binance_ohlcv"])

        print("5/8 Construccion de market_daily + data_quality")
        details["market_daily"] = build_market_daily()
        print({k: v for k, v in details["market_daily"].items() if k != "summary"})

        print("6/8 Noticias + sentimiento")
        details["news"] = refresh_news()
        print(details["news"])

        print("7/8 Entrenamiento XGBoost + walk-forward")
        trained = train_all()
        details["models_trained"] = sum(x.get("status") == "trained" for x in trained)
        details["model_status"] = [{"coin_id": x.get("coin_id"), "status": x.get("status"), "rows": x.get("rows")} for x in trained]
        print("Modelos entrenados:", details["models_trained"], "/", len(trained))
        skipped_models = [x for x in details["model_status"] if x.get("status") != "trained"]
        if skipped_models:
            print("No entrenados/omitidos:", skipped_models)

        print(f"8/8 Forecast configurable ({settings.forecast_days} dias)")
        forecasts = forecast_all()
        forecasted = [x for x in forecasts if x.get("status") == "forecasted"]
        details["forecasts"] = len(forecasted)
        details["forecast_points"] = sum(int(x.get("days") or 0) for x in forecasted)
        details["forecast_status"] = forecasts
        print("Activos con forecast:", details["forecasts"], "/", len(forecasts))
        print("Puntos futuros generados:", details["forecast_points"])
        skipped_forecasts = [
            {"coin_id": x.get("coin_id"), "status": x.get("status"), "horizon": x.get("horizon")}
            for x in forecasts if x.get("status") != "forecasted"
        ]
        if skipped_forecasts:
            print("Sin forecast:", skipped_forecasts)

        print("Publicando capa plana para Power BI")
        details["powerbi"] = publish_powerbi_layer()
        print(details["powerbi"])

        log_run("bootstrap", "success", started, details)
        print("Bootstrap finalizado.")
    except Exception as exc:
        log_run("bootstrap", "error", started, details, str(exc))
        raise
