from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pymongo import UpdateOne
from src.api_clients.mobula import MobulaClient
from src.api_clients.binance import BinanceClient
from src.config import settings
from src.db import get_db

DAY_MS = 86_400_000


def _utc_day(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def load_price_history(days=None):
    """Carga RAW de precio historico Mobula Asset History."""
    if not settings.mobula_api_key:
        return {"assets": 0, "rows": 0, "warning": "MOBULA_API_KEY ausente"}
    db = get_db()
    days = int(days or settings.history_days)
    assets = list(db.assets.find(
        {"active_top_n": True, "mobula_id": {"$ne": None}},
        {"_id": 0, "coin_id": 1, "mobula_id": 1, "symbol": 1}
    ).sort("rank", 1))
    if not assets:
        return {"assets": 0, "rows": 0, "failed": []}

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    mapping = {int(a["mobula_id"]): a["coin_id"] for a in assets}
    payload = [{
        "id": int(a["mobula_id"]),
        "period": "1d",
        "from": int(start.timestamp() * 1000),
        "to": int(_utc_day(now).timestamp() * 1000) - 1,
    } for a in assets]

    responses = MobulaClient().price_history_batch(payload)
    ops, covered, failures = [], set(), []
    seen_ids = set()
    for item in responses:
        mid = item.get("id")
        if mid is not None:
            try:
                seen_ids.add(int(mid))
            except Exception:
                pass
        if item.get("error") or mid is None:
            failures.append({"mobula_id": mid, "error": str(item.get("error") or "missing_id")})
            continue
        coin_id = mapping.get(int(mid))
        if not coin_id:
            continue
        rows = item.get("priceHistory", []) or []
        if not rows:
            failures.append({"coin_id": coin_id, "mobula_id": int(mid), "error": "empty_price_history"})
            continue
        covered.add(coin_id)
        for point in rows:
            if not isinstance(point, (list, tuple)) or len(point) < 2 or point[1] is None:
                continue
            dt = datetime.fromtimestamp(float(point[0]) / 1000, tz=timezone.utc)
            date = _utc_day(dt)
            doc = {
                "coin_id": coin_id,
                "mobula_id": int(mid),
                "date": date,
                "price": float(point[1]),
                "source": "mobula_asset_price_history",
                "updated_at": now,
            }
            ops.append(UpdateOne(
                {"coin_id": coin_id, "date": date, "source": doc["source"]},
                {"$set": doc}, upsert=True
            ))
    if ops:
        db.price_history.bulk_write(ops, ordered=False)

    for mid, coin_id in mapping.items():
        if mid not in seen_ids:
            failures.append({"coin_id": coin_id, "mobula_id": mid, "error": "no_response_from_batch"})
    return {"assets": len(covered), "rows": len(ops), "failed": failures}


def best_token(asset):
    tokens = asset.get("mobula_tokens", []) or []
    candidates = [t for t in tokens if t.get("address") and (t.get("chainId") or t.get("blockchain"))]
    if not candidates:
        return None

    def val(t):
        try:
            return float(t.get("liquidityUSD") or 0), float(t.get("volume24hUSD") or 0)
        except Exception:
            return (0.0, 0.0)
    return sorted(candidates, key=val, reverse=True)[0]


def load_ohlcv(days=None):
    """Carga RAW OHLCV Mobula para activos contract-based."""
    if not settings.mobula_api_key:
        return {"assets": 0, "rows": 0, "warning": "MOBULA_API_KEY ausente"}
    db = get_db()
    days = int(days or settings.history_days)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    assets = list(db.assets.find(
        {"active_top_n": True},
        {"_id": 0, "coin_id": 1, "mobula_tokens": 1}
    ).sort("rank", 1))

    payload, mapping, skipped = [], {}, []
    for a in assets:
        token = best_token(a)
        if not token:
            skipped.append(a["coin_id"])
            continue
        address = token["address"]
        chain = token.get("chainId") or token.get("blockchain")
        mapping[str(address).lower()] = {"coin_id": a["coin_id"], "token": token}
        payload.append({
            "address": address,
            "chainId": chain,
            "period": "1d",
            "from": int(start.timestamp() * 1000),
            "to": int(_utc_day(now).timestamp() * 1000) - 1,
            "amount": min(days + 10, 2000),
            "usd": True,
            "fill": True,
            "enableAssetHistory": True,
        })

    if not payload:
        return {"assets": 0, "rows": 0, "skipped": skipped, "failed": []}

    responses = MobulaClient().token_ohlcv_batch(payload)
    ops, covered, failed = [], set(), []
    for item in responses:
        if item.get("error"):
            failed.append({"address": item.get("address"), "error": str(item.get("error"))})
            continue
        meta = mapping.get(str(item.get("address", "")).lower())
        if not meta:
            continue
        coin_id = meta["coin_id"]
        rows = item.get("ohlcv", []) or []
        if not rows:
            failed.append({"coin_id": coin_id, "error": "empty_ohlcv"})
            continue
        covered.add(coin_id)
        for c in rows:
            if c.get("t") is None:
                continue
            ts = _utc_day(datetime.fromtimestamp(float(c["t"]) / 1000, tz=timezone.utc))
            doc = {
                "coin_id": coin_id,
                "timestamp": ts,
                "period": "1d",
                "open": c.get("o"), "high": c.get("h"), "low": c.get("l"),
                "close": c.get("c"), "volume": c.get("v"),
                "source": "mobula_token_ohlcv",
                "source_token_address": meta["token"].get("address"),
                "source_chain_id": meta["token"].get("chainId") or meta["token"].get("blockchain"),
                "updated_at": now,
            }
            ops.append(UpdateOne(
                {"coin_id": coin_id, "timestamp": ts, "period": "1d", "source": doc["source"]},
                {"$set": doc}, upsert=True
            ))
    if ops:
        db.ohlcv.bulk_write(ops, ordered=False)
    return {"assets": len(covered), "rows": len(ops), "skipped": skipped, "failed": failed}


def load_binance_ohlcv(days=None):
    """Fallback RAW OHLCV desde Binance Spot (publico, sin API key)."""
    db = get_db()
    days = int(days or settings.history_days)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    assets = list(db.assets.find(
        {"active_top_n": True},
        {"_id": 0, "coin_id": 1, "symbol": 1}
    ).sort("rank", 1))

    client = BinanceClient()
    ops, covered, failed = [], set(), []
    for a in assets:
        symbol = str(a.get("symbol") or "").upper()
        if not symbol or symbol in {"USDT", "USDC", "FDUSD", "DAI", "USDE", "USDS", "TUSD"}:
            continue
        pair = f"{symbol}USDT"
        try:
            rows = client.klines(pair, int(start.timestamp() * 1000), int(_utc_day(now).timestamp() * 1000) - 1, "1d")
            if not rows:
                failed.append({"coin_id": a["coin_id"], "pair": pair, "error": "empty_or_pair_unavailable"})
                continue
            covered.add(a["coin_id"])
            for row in rows:
                k = client.normalize_kline(row)
                ts = _utc_day(k["timestamp"])
                doc = {
                    "coin_id": a["coin_id"],
                    "symbol": symbol,
                    "pair": pair,
                    "timestamp": ts,
                    "period": "1d",
                    "open": k["open"], "high": k["high"], "low": k["low"],
                    "close": k["close"], "volume": k["volume"],
                    "quote_volume": k["quote_volume"], "trades": k["trades"],
                    "source": "binance_spot_1d",
                    "updated_at": now,
                }
                ops.append(UpdateOne(
                    {"coin_id": a["coin_id"], "timestamp": ts, "period": "1d", "source": doc["source"]},
                    {"$set": doc}, upsert=True
                ))
        except Exception as exc:
            failed.append({"coin_id": a["coin_id"], "pair": pair, "error": str(exc)[:500]})
    if ops:
        db.binance_ohlcv.bulk_write(ops, ordered=False)
    return {"assets": len(covered), "rows": len(ops), "failed": failed}


def build_market_daily(days=None):
    """Construye la serie canonica diaria usada por ML y Power BI.

    Prioridad por dia: Mobula OHLCV > Binance OHLCV > Mobula price history.
    Los datos faltantes no se inventan.
    """
    db = get_db()
    days = int(days or settings.history_days)
    now = datetime.now(timezone.utc)
    start_day = _utc_day(now - timedelta(days=days))
    assets = list(db.assets.find(
        {"active_top_n": True}, {"_id": 0, "coin_id": 1, "symbol": 1, "name": 1, "rank": 1}
    ).sort("rank", 1))

    total_ops = 0
    summaries = []
    for asset in assets:
        coin_id = asset["coin_id"]
        by_date = {}

        # 1) price history: base de cierre, cobertura amplia
        for r in db.price_history.find(
            {"coin_id": coin_id, "date": {"$gte": start_day}},
            {"_id": 0, "date": 1, "price": 1, "source": 1}
        ).sort("date", 1):
            if r.get("price") is None:
                continue
            d = _utc_day(r["date"])
            if d >= _utc_day(now):
                continue
            by_date[d] = {
                "date": d, "open": None, "high": None, "low": None,
                "close": float(r["price"]), "volume": None, "quote_volume": None,
                "source": "mobula_asset_price_history", "source_priority": 3,
            }

        # 2) Binance OHLCV: reemplaza cuando hay OHLCV completo
        for r in db.binance_ohlcv.find(
            {"coin_id": coin_id, "timestamp": {"$gte": start_day}, "period": "1d"},
            {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1,
             "volume": 1, "quote_volume": 1, "source": 1, "pair": 1}
        ).sort("timestamp", 1):
            if r.get("close") is None:
                continue
            d = _utc_day(r["timestamp"])
            if d >= _utc_day(now):
                continue
            by_date[d] = {
                "date": d, "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r.get("volume") or 0),
                "quote_volume": float(r.get("quote_volume") or 0),
                "source": "binance_spot_1d", "source_priority": 2,
                "source_pair": r.get("pair"),
            }

        # 3) Mobula token OHLCV: prioridad mas alta cuando existe
        for r in db.ohlcv.find(
            {"coin_id": coin_id, "timestamp": {"$gte": start_day}, "period": "1d"},
            {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1,
             "volume": 1, "source": 1}
        ).sort("timestamp", 1):
            if r.get("close") is None:
                continue
            d = _utc_day(r["timestamp"])
            if d >= _utc_day(now):
                continue
            by_date[d] = {
                "date": d,
                "open": float(r["open"]) if r.get("open") is not None else None,
                "high": float(r["high"]) if r.get("high") is not None else None,
                "low": float(r["low"]) if r.get("low") is not None else None,
                "close": float(r["close"]),
                "volume": float(r.get("volume") or 0) if r.get("volume") is not None else None,
                "quote_volume": None,
                "source": "mobula_token_ohlcv", "source_priority": 1,
            }

        ops = []
        source_counts = {}
        for d, row in sorted(by_date.items()):
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
            doc = {
                "coin_id": coin_id,
                "symbol": asset.get("symbol"),
                "name": asset.get("name"),
                "rank": asset.get("rank"),
                **row,
                "updated_at": now,
            }
            ops.append(UpdateOne(
                {"coin_id": coin_id, "date": d}, {"$set": doc}, upsert=True
            ))
        # Una vela de hoy sigue abierta; nunca entra a entrenamiento ni al dashboard histórico.
        db.market_daily.delete_many({"coin_id": coin_id, "date": {"$gte": _utc_day(now)}})
        if ops:
            db.market_daily.bulk_write(ops, ordered=False)
            total_ops += len(ops)

        dates = sorted(by_date)
        if dates:
            first_date, last_date = dates[0], dates[-1]
            life_expected = (last_date - first_date).days + 1
            requested_expected = (min(_utc_day(now), last_date) - max(start_day, first_date)).days + 1
            expected = max(1, requested_expected)
            requested_window_expected = max(1, (_utc_day(now) - start_day).days)
            actual = len(dates)
            coverage = min(100.0, actual / expected * 100.0)
            requested_coverage = min(100.0, actual / requested_window_expected * 100.0)
        else:
            first_date = last_date = None
            life_expected = 0
            expected = days + 1
            requested_window_expected = days
            actual = 0
            coverage = 0.0
            requested_coverage = 0.0

        quality = {
            "coin_id": coin_id,
            "symbol": asset.get("symbol"),
            "checked_at": now,
            "requested_start": start_day,
            "requested_end": _utc_day(now),
            "first_available_date": first_date,
            "last_available_date": last_date,
            "expected_rows": expected,
            "actual_rows": actual,
            "missing_rows": max(0, expected - actual),
            "coverage_pct": float(round(coverage, 4)),
            "requested_window_coverage_pct": float(round(requested_coverage, 4)),
            "source_counts": source_counts,
            "status": "OK" if requested_coverage >= 95 else ("WARNING" if requested_coverage >= 80 else "LOW_COVERAGE"),
        }
        db.data_quality.insert_one(quality)
        db.assets.update_one({"coin_id": coin_id}, {"$set": {
            "history.first_date": first_date,
            "history.last_date": last_date,
            "history.rows": actual,
            "history.coverage_pct": float(round(requested_coverage, 4)),
            "history.updated_at": now,
        }})
        summaries.append({"coin_id": coin_id, "rows": actual, "coverage_pct": round(coverage, 2), "sources": source_counts})

    return {"assets": len(assets), "rows_upserted": total_ops, "summary": summaries}
