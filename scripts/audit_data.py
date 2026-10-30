"""Audita la base real sin modificarla ni imprimir credenciales.

Uso: python scripts/audit_data.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collections import Counter
from datetime import datetime, timezone, timedelta
import math
from src.db import get_db, ping
from src.config import settings


def audit():
    ping()
    db = get_db()
    now = datetime.now(timezone.utc)
    assets = list(db.assets.find({"active_top_n": True}, {"_id": 0, "coin_id": 1, "symbol": 1, "rank": 1}))
    print(f"Base: {settings.mongodb_db_name} | Activos activos: {len(assets)}")
    print("Moneda | filas | primer día | último día | cobertura ventana | huecos | inválidos | spot/cierre % | fuente | modelo | forecast")
    critical = 0
    for asset in sorted(assets, key=lambda a: a.get("rank") or 999):
        cid = asset["coin_id"]
        rows = list(db.market_daily.find({"coin_id": cid}, {"_id": 0, "date": 1, "close": 1, "source": 1}).sort("date", 1))
        dates = [r["date"].date() for r in rows if isinstance(r.get("date"), datetime)]
        dup = len(dates) - len(set(dates))
        gaps = sum((b-a).days-1 for a,b in zip(dates,dates[1:]) if (b-a).days > 1)
        invalid = sum(not isinstance(r.get("close"),(int,float)) or not math.isfinite(r["close"]) or r["close"] <= 0 for r in rows)
        future = sum(d >= now.date() for d in dates)
        source = Counter(r.get("source") or "?" for r in rows)
        start = (now - timedelta(days=settings.history_days)).date()
        coverage = sum(d >= start and d < now.date() for d in set(dates)) / settings.history_days * 100
        metric = db.model_metrics.find_one({"coin_id": cid}, sort=[("trained_at",-1)])
        forecast = db.forecast_runs.find_one({"coin_id": cid, "is_latest": True})
        snapshot = db.market_snapshots.find_one({"coin_id": cid}, sort=[("observed_at", -1)])
        delta = None
        if rows and snapshot and isinstance(snapshot.get("price"), (int, float)) and snapshot["price"] > 0 and isinstance(rows[-1].get("close"), (int, float)) and rows[-1]["close"] > 0:
            delta = (snapshot["price"] / rows[-1]["close"] - 1) * 100
        model = "N/A" if not metric else ("mejor" if metric.get("beats_zero_return_baseline") else "NO supera baseline")
        forecast_days = (forecast or {}).get("forecast_days", 0)
        print(f"{asset.get('symbol','?'):7} | {len(rows):5} | {str(dates[0]) if dates else '-':10} | {str(dates[-1]) if dates else '-':10} | {coverage:5.1f}% | {gaps:5} | {invalid+future+dup:7} | {f'{delta:+.1f}%' if delta is not None else '-':13} | {source.most_common(1)[0][0] if source else '-':24} | {model:18} | {forecast_days}")
        if dup or invalid or future:
            print(f"  PROBLEMA: duplicados={dup}, cierres inválidos={invalid}, fechas futuras={future}")
            critical += 1
        if dates and (now.date()-dates[-1]).days > 3:
            print("  AVISO: histórico sin actualizar en más de 3 días")
        elif delta is not None and abs(delta) > 20:
            print("  PROBLEMA: el precio spot difiere más de 20% del último cierre diario")
            critical += 1
        if coverage < 80:
            print("  AVISO: cobertura menor al 80% de la ventana solicitada")
        if forecast and metric and forecast.get("model_version") != metric.get("model_version"):
            print("  AVISO: pronóstico y métrica corresponden a versiones distintas")
    print(f"Activos con problemas críticos: {critical}")
    return critical


if __name__ == "__main__":
    sys.exit(1 if audit() else 0)
