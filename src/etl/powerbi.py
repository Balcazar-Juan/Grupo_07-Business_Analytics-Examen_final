from __future__ import annotations
from datetime import datetime, timezone
from pymongo import UpdateOne
from src.db import get_db


def publish_dim_assets():
    """Publica una dimension plana y estable para Power BI.

    No expone objetos/arrays anidados de ``assets`` (providers, mobula_asset,
    mobula_tokens, history), que pueden generar problemas de inferencia de
    esquema en Atlas SQL/Power BI.
    """
    db = get_db()
    now = datetime.now(timezone.utc)
    rows = list(db.assets.find({"active_top_n": True}).sort("rank", 1))
    ops = []
    for a in rows:
        providers = a.get("providers") or {}
        coinstats = providers.get("coinstats") or {}
        mobula = providers.get("mobula") or {}
        binance = providers.get("binance") or {}
        history = a.get("history") or {}
        doc = {
            "coin_id": a.get("coin_id"),
            "name": a.get("name"),
            "symbol": a.get("symbol"),
            "rank": a.get("rank"),
            "active_top_n": bool(a.get("active_top_n", False)),
            "is_stablecoin": bool(a.get("is_stablecoin", False)),
            "coinstats_id": coinstats.get("id") or a.get("coin_id"),
            "mobula_id": mobula.get("id") if mobula.get("id") is not None else a.get("mobula_id"),
            "binance_pair": binance.get("pair"),
            "history_first_date": history.get("first_date"),
            "history_last_date": history.get("last_date"),
            "history_rows": int(history.get("rows") or 0),
            "history_coverage_pct": float(history.get("coverage_pct") or 0.0),
            "updated_at": now,
        }
        if doc["coin_id"]:
            ops.append(UpdateOne({"coin_id": doc["coin_id"]}, {"$set": doc}, upsert=True))
    if ops:
        db.dim_assets.bulk_write(ops, ordered=False)
    # Elimina activos que ya no pertenecen al universo actual para que la dimension
    # represente exactamente el Top N vigente.
    active_ids = [x.get("coin_id") for x in rows if x.get("coin_id")]
    if active_ids:
        db.dim_assets.delete_many({"coin_id": {"$nin": active_ids}})
    return {"rows": len(ops)}


def publish_data_quality_bi():
    """Aplana el ultimo control de calidad por activo para consumo BI."""
    db = get_db()
    assets = list(db.assets.find({"active_top_n": True}, {"_id": 0, "coin_id": 1}))
    ops = []
    for a in assets:
        coin_id = a["coin_id"]
        q = db.data_quality.find_one({"coin_id": coin_id}, sort=[("checked_at", -1)])
        if not q:
            continue
        sources = q.get("source_counts") or {}
        doc = {
            "coin_id": coin_id,
            "symbol": q.get("symbol"),
            "checked_at": q.get("checked_at"),
            "requested_start": q.get("requested_start"),
            "requested_end": q.get("requested_end"),
            "first_available_date": q.get("first_available_date"),
            "last_available_date": q.get("last_available_date"),
            "expected_rows": int(q.get("expected_rows") or 0),
            "actual_rows": int(q.get("actual_rows") or 0),
            "missing_rows": int(q.get("missing_rows") or 0),
            "coverage_pct": float(q.get("coverage_pct") or 0.0),
            "requested_window_coverage_pct": float(q.get("requested_window_coverage_pct") or 0.0),
            "rows_mobula_price_history": int(sources.get("mobula_asset_price_history") or 0),
            "rows_binance_ohlcv": int(sources.get("binance_spot_1d") or 0),
            "rows_mobula_ohlcv": int(sources.get("mobula_token_ohlcv") or 0),
            "status": q.get("status"),
        }
        ops.append(UpdateOne({"coin_id": coin_id}, {"$set": doc}, upsert=True))
    if ops:
        db.data_quality_bi.bulk_write(ops, ordered=False)
    return {"rows": len(ops)}


def publish_news_bi():
    """Publica noticias sin arrays y una tabla puente noticia-activo."""
    db = get_db()
    news_ops = []
    bridge_ops = []
    for n in db.news.find({}):
        news_id = n.get("news_id")
        if not news_id:
            continue
        doc = {
            "news_id": news_id,
            "source": n.get("source"),
            "title": n.get("title"),
            "summary": n.get("summary"),
            "url": n.get("url"),
            "published_at": n.get("published_at"),
            "ingested_at": n.get("ingested_at"),
            "sentiment_compound": n.get("sentiment_compound"),
            "sentiment_pos": n.get("sentiment_pos"),
            "sentiment_neu": n.get("sentiment_neu"),
            "sentiment_neg": n.get("sentiment_neg"),
            "sentiment_label": n.get("sentiment_label"),
        }
        news_ops.append(UpdateOne({"news_id": news_id}, {"$set": doc}, upsert=True))
        for coin_id in (n.get("mentioned_coin_ids") or []):
            bridge_ops.append(UpdateOne(
                {"news_id": news_id, "coin_id": coin_id},
                {"$set": {"news_id": news_id, "coin_id": coin_id, "published_at": n.get("published_at")}},
                upsert=True,
            ))
    if news_ops:
        db.news_bi.bulk_write(news_ops, ordered=False)
    if bridge_ops:
        db.bridge_news_assets.bulk_write(bridge_ops, ordered=False)
    return {"news_rows": len(news_ops), "bridge_rows": len(bridge_ops)}


def publish_model_metrics_bi():
    """Aplana la ultima metrica y separa la importancia de variables."""
    db = get_db()
    assets = list(db.assets.find({"active_top_n": True}, {"_id": 0, "coin_id": 1}))
    metric_ops = []
    importance_ops = []
    for a in assets:
        coin_id = a["coin_id"]
        m = db.model_metrics.find_one({"coin_id": coin_id}, sort=[("trained_at", -1)])
        if not m:
            continue
        version = m.get("model_version")
        doc = {
            "coin_id": coin_id,
            "symbol": m.get("symbol"),
            "trained_at": m.get("trained_at"),
            "model_version": version,
            "history_source": m.get("history_source"),
            "training_start": m.get("training_start"),
            "training_end": m.get("training_end"),
            "training_rows": int(m.get("training_rows") or 0),
            "target": m.get("target"),
            "mae_log_return": m.get("mae_log_return"),
            "rmse_log_return": m.get("rmse_log_return"),
            "direction_accuracy_pct": m.get("direction_accuracy_pct"),
            "baseline_mae_log_return": m.get("baseline_mae_log_return"),
            "beats_zero_return_baseline": m.get("beats_zero_return_baseline"),
            "residual_std": m.get("residual_std"),
            "forecast_return_clip_low": m.get("forecast_return_clip_low"),
            "forecast_return_clip_high": m.get("forecast_return_clip_high"),
        }
        metric_ops.append(UpdateOne({"coin_id": coin_id}, {"$set": doc}, upsert=True))
        for feature, importance in (m.get("feature_importance") or {}).items():
            importance_ops.append(UpdateOne(
                {"coin_id": coin_id, "model_version": version, "feature": feature},
                {"$set": {
                    "coin_id": coin_id,
                    "symbol": m.get("symbol"),
                    "model_version": version,
                    "trained_at": m.get("trained_at"),
                    "feature": feature,
                    "importance": float(importance),
                }},
                upsert=True,
            ))
    if metric_ops:
        db.model_metrics_bi.bulk_write(metric_ops, ordered=False)
    if importance_ops:
        db.model_feature_importance.bulk_write(importance_ops, ordered=False)
    return {"metric_rows": len(metric_ops), "importance_rows": len(importance_ops)}


def publish_powerbi_layer():
    return {
        "dim_assets": publish_dim_assets(),
        "data_quality_bi": publish_data_quality_bi(),
        "news_bi": publish_news_bi(),
        "model_metrics_bi": publish_model_metrics_bi(),
    }
