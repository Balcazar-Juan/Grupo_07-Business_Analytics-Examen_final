from __future__ import annotations
from datetime import datetime, timedelta, timezone
import math
import os
from pymongo import UpdateOne
from src.db import get_db
from src.api_clients.mobula import MobulaClient
from src.api_clients.binance import BinanceClient
from src.api_clients.coingecko import CoinGeckoClient

DAY_MS = 86_400_000


def day_utc(value):
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def valid_price(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def refresh_historical_fallback(days=14):
    db = get_db()
    now = datetime.now(timezone.utc)
    today = day_utc(now)
    days = min(max(int(days), 1), 365)
    start = today - timedelta(days=days)
    required = {start + timedelta(days=i) for i in range(days)}
    assets = list(db.assets.find({"active_top_n": True},
        {"_id": 0, "coin_id": 1, "symbol": 1, "name": 1, "mobula_id": 1,
         "providers": 1, "is_stablecoin": 1}).sort("rank", 1))
    mobula = None
    cg = CoinGeckoClient() if os.getenv("COINGECKO_API_KEY", "").strip() else None
    binance = BinanceClient()
    mobula_denied = False
    summary = []

    for asset in assets:
        cid = asset["coin_id"]
        symbol = (asset.get("symbol") or "").upper()
        attempts = []
        accepted = {}  
        price_ops = []
        binance_ops = []

        # 1. MOBULA
        try:
            if mobula_denied:
                raise RuntimeError("Mobula deshabilitada tras HTTP 401/403 en esta ejecución")
            if mobula is None:
                mobula = MobulaClient()
            mid = asset.get("mobula_id")
            if mid is None:
                match = mobula.search_asset(symbol, asset.get("name"))
                if match and match.get("id") is not None:
                    mid = int(match["id"])
                    db.assets.update_one({"coin_id": cid}, {"$set": {
                        "mobula_id": mid, "providers.mobula.id": mid,
                        "mobula_resolved_at": now}})
            if mid is None:
                raise ValueError("Sin mobula_id; búsqueda sin coincidencia")
            response = mobula.price_history_batch([{
                "id": int(mid), "period": "1d", "from": int(start.timestamp()*1000),
                "to": int(today.timestamp()*1000)-1
            }])
            if not response or response[0].get("error"):
                raise ValueError(str(response[0].get("error") if response else "respuesta vacía"))
            points = response[0].get("priceHistory", []) or []
            for point in points:
                if len(point) < 2 or not valid_price(point[1]):
                    continue
                d = day_utc(datetime.fromtimestamp(float(point[0])/1000, timezone.utc))
                if d not in required:
                    continue
                accepted[d] = "mobula_asset_price_history"
                price_ops.append(UpdateOne({"coin_id": cid, "date": d,
                    "source": "mobula_asset_price_history"}, {"$set": {
                    "coin_id": cid, "mobula_id": int(mid), "date": d,
                    "price": float(point[1]), "source": "mobula_asset_price_history",
                    "updated_at": now}}, upsert=True))
            attempts.append({"source": "mobula", "dates": sum(v == "mobula_asset_price_history" for v in accepted.values())})
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
                mobula_denied = True
            attempts.append({"source": "mobula", "error": str(exc)[:180]})

        # 2. COINGECKO
        missing = required - accepted.keys()
        if missing and cg is not None:
            try:
                cg_id = ((asset.get("providers") or {}).get("coingecko") or {}).get("id") or cid
                try:
                    points = cg.historical_prices(cg_id, days=min(days+2, 365))
                except Exception as exc:
                    if getattr(getattr(exc, "response", None), "status_code", None) != 404:
                        raise
                    cg_id = cg.find_id(cid, symbol, asset.get("name"))
                    if not cg_id:
                        raise ValueError("CoinGecko ID no encontrado con coincidencia exacta")
                    points = cg.historical_prices(cg_id, days=min(days+2, 365))
                if points:
                    db.assets.update_one({"coin_id": cid}, {"$set": {"providers.coingecko.id": cg_id}})
                added = 0
                for point in points:
                    if len(point) < 2 or not valid_price(point[1]):
                        continue
                    d = day_utc(datetime.fromtimestamp(float(point[0])/1000, timezone.utc))
                    if d not in missing:
                        continue
                    accepted[d] = "coingecko_market_chart"
                    price_ops.append(UpdateOne({"coin_id": cid, "date": d,
                        "source": "coingecko_market_chart"}, {"$set": {
                        "coin_id": cid, "date": d, "price": float(point[1]),
                        "source": "coingecko_market_chart", "coingecko_id": cg_id,
                        "updated_at": now}}, upsert=True))
                    added += 1
                attempts.append({"source": "coingecko", "dates": added})
            except Exception as exc:
                attempts.append({"source": "coingecko", "error": str(exc)[:180]})
        elif missing:
            attempts.append({"source": "coingecko", "error": "COINGECKO_API_KEY ausente"})

        # 3. BINANCE
        missing = required - accepted.keys()
        if missing and symbol and symbol not in {"USDT", "USDC", "DAI", "FDUSD", "USDE", "USDS"}:
            try:
                pair = ((asset.get("providers") or {}).get("binance") or {}).get("pair") or f"{symbol}USDT"
                rows = binance.klines(pair, int(start.timestamp()*1000), int(today.timestamp()*1000)-1)
                added = 0
                for row in rows:
                    k = binance.normalize_kline(row)
                    d = day_utc(k["timestamp"])
                    if d not in missing or not valid_price(k["close"]):
                        continue
                    accepted[d] = "binance_spot_1d"
                    binance_ops.append(UpdateOne({"coin_id": cid, "timestamp": d,
                        "period": "1d", "source": "binance_spot_1d"}, {"$set": {
                        "coin_id": cid, "symbol": symbol, "pair": pair,
                        "timestamp": d, "period": "1d", "open": k["open"],
                        "high": k["high"], "low": k["low"], "close": k["close"],
                        "volume": k["volume"], "quote_volume": k["quote_volume"],
                        "trades": k["trades"], "source": "binance_spot_1d",
                        "updated_at": now}}, upsert=True))
                    added += 1
                attempts.append({"source": "binance", "dates": added})
            except Exception as exc:
                attempts.append({"source": "binance", "error": str(exc)[:180]})

        if price_ops:
            db.price_history.bulk_write(price_ops, ordered=False)
        if binance_ops:
            db.binance_ohlcv.bulk_write(binance_ops, ordered=False)

        missing = required - accepted.keys()
        item = {"coin_id": cid, "dates_received": len(accepted),
                "missing_dates": len(missing), "last_received": max(accepted).isoformat() if accepted else None,
                "attempts": attempts}
        print("HISTORICAL", item, flush=True)
        summary.append(item)

    return {"assets": len(summary), "with_new_data": sum(x["dates_received"] > 0 for x in summary),
            "without_data": [x["coin_id"] for x in summary if not x["dates_received"]],
            "summary": summary}
