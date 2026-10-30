from datetime import datetime, timezone
from pymongo import UpdateOne
from src.api_clients.coinstats import CoinStatsClient
from src.api_clients.mobula import MobulaClient
from src.config import settings
from src.db import get_db
from requests import HTTPError

def refresh_coinstats():
    db = get_db()
    now = datetime.now(timezone.utc)
    coins = CoinStatsClient().top_coins(settings.top_n)
    if not isinstance(coins, list) or not coins:
        raise ValueError("CoinStats devolvió un catálogo vacío o inválido; se conserva el Top N anterior")
    ops, snapshots, active_ids = [], [], []

    for coin in coins:
        coin_id = coin.get("id")
        if not coin_id:
            continue
        active_ids.append(coin_id)
        ops.append(UpdateOne(
            {"coin_id": coin_id},
            {
                "$set": {
                    "coin_id": coin_id,
                    "name": coin.get("name"),
                    "symbol": coin.get("symbol"),
                    "rank": coin.get("rank"),
                    "icon": coin.get("icon"),
                    "active_top_n": True,
                    "providers.coinstats.id": coin_id,
                    "providers.binance.pair": f"{str(coin.get('symbol') or '').upper()}USDT",
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        ))
        snapshots.append({
            "coin_id": coin_id,
            "name": coin.get("name"),
            "symbol": coin.get("symbol"),
            "rank": coin.get("rank"),
            "price": coin.get("price"),
            "market_cap": coin.get("marketCap"),
            "volume_24h": coin.get("volume"),
            "price_change_1h": coin.get("priceChange1h"),
            "price_change_1d": coin.get("priceChange1d"),
            "price_change_1w": coin.get("priceChange1w"),
            "price_change_1m": coin.get("priceChange1m"),
            "source": "coinstats",
            "observed_at": now,
        })
    if ops:
        db.assets.bulk_write(ops)
        # Solo se desactivan activos anteriores después de persistir un catálogo válido.
        db.assets.update_many(
            {"coin_id": {"$nin": active_ids}},
            {"$set": {"active_top_n": False}},
        )
    if snapshots:
        db.market_snapshots.insert_many(snapshots)
    return len(snapshots)

def resolve_mobula():
    db = get_db()
    client = MobulaClient()
    rows = list(db.assets.find(
        {"active_top_n": True},
        {"_id": 0, "coin_id": 1, "symbol": 1, "name": 1, "mobula_id": 1}
    ).sort("rank", 1))
    resolved, failed = 0, []
    for a in rows:
        if a.get("mobula_id") is not None:
            continue
        try:
            match = client.search_asset(a["symbol"], a.get("name"))
            if not match or match.get("id") is None:
                failed.append(a["coin_id"])
                continue
            details = client.asset_details(match["id"])
            asset = details.get("asset", {}) or {}
            db.assets.update_one(
                {"coin_id": a["coin_id"]},
                {"$set": {
                    "mobula_id": int(match["id"]),
                    "providers.mobula.id": int(match["id"]),
                    "mobula_asset": asset,
                    "mobula_tokens": details.get("tokens", []) or [],
                    "is_stablecoin": bool(asset.get("isStablecoin", False)),
                    "mobula_native_chain_id": asset.get("nativeChainId"),
                    "mobula_resolved_at": datetime.now(timezone.utc),
                }}
            )
            resolved += 1
        except Exception as exc:
            # Una clave sin permisos no se soluciona repitiendo 20 peticiones.
            if isinstance(exc, HTTPError):
                response = exc.response
                if response is not None and response.status_code in (401, 403):
                    raise
            failed.append(a["coin_id"])
    return {"resolved": resolved, "failed": failed}

def refresh_mobula():
    db = get_db()
    client = MobulaClient()
    assets = list(db.assets.find(
        {"active_top_n": True, "mobula_id": {"$ne": None}},
        {"_id": 0, "coin_id": 1, "mobula_id": 1}
    ))
    if not assets:
        return 0
    mapping = {int(x["mobula_id"]): x["coin_id"] for x in assets}
    details = client.asset_details_batch(list(mapping.keys()))
    now = datetime.now(timezone.utc)
    docs = []
    for item in details:
        asset = item.get("asset", {}) or {}
        mid = asset.get("id")
        if mid is None or int(mid) not in mapping:
            continue
        coin_id = mapping[int(mid)]
        db.assets.update_one({"coin_id": coin_id}, {"$set": {
            "mobula_asset": asset,
            "mobula_tokens": item.get("tokens", []) or [],
            "is_stablecoin": bool(asset.get("isStablecoin", False)),
            "mobula_updated_at": now,
        }})
        docs.append({
            "coin_id": coin_id,
            "mobula_id": int(mid),
            "name": asset.get("name"),
            "symbol": asset.get("symbol"),
            "rank": asset.get("rank"),
            "price": asset.get("priceUSD"),
            "market_cap": asset.get("marketCapUSD"),
            "is_stablecoin": bool(asset.get("isStablecoin", False)),
            "source": "mobula",
            "observed_at": now,
        })
    if docs:
        db.mobula_snapshots.insert_many(docs)
    return len(docs)

def refresh_market():
    count = refresh_coinstats()
    if not settings.mobula_api_key:
        return {"coinstats": count, "mobula": 0, "resolution": {"resolved": 0, "warning": "MOBULA_API_KEY ausente"}}
    try:
        resolution = resolve_mobula()
        mobula = refresh_mobula()
    except HTTPError as exc:
        if exc.response is None or exc.response.status_code not in (401, 403):
            raise
        resolution = {"resolved": 0, "warning": "Mobula no autorizado (HTTP 401/403)"}
        mobula = 0
    return {"coinstats": count, "mobula": mobula, "resolution": resolution}
