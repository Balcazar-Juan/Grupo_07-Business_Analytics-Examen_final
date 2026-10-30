from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import ensure_indexes
from src.ml.model import train_all, forecast_all
from src.config import settings

if __name__ == "__main__":
    ensure_indexes()
    trained = train_all()
    print("Entrenamiento:", [{"coin_id": x.get("coin_id"), "status": x.get("status")} for x in trained])
    forecasts = forecast_all()
    print(f"Forecast ({settings.forecast_days} dias):", forecasts)
