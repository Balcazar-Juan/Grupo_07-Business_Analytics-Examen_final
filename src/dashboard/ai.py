from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os

from src.db import get_db


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "postura": {"type": "STRING", "enum": ["OBSERVAR", "ESPERAR", "EVALUAR_COMPRA", "EVALUAR_VENTA"]},
        "riesgo": {"type": "STRING", "enum": ["BAJO", "MEDIO", "ALTO"]},
        "resumen": {"type": "STRING"},
        "escenario_alcista": {"type": "STRING"},
        "escenario_base": {"type": "STRING"},
        "escenario_bajista": {"type": "STRING"},
        "senales_a_vigilar": {"type": "ARRAY", "items": {"type": "STRING"}},
        "limitaciones": {"type": "STRING"},
    },
    "required": ["postura", "riesgo", "resumen", "escenario_alcista", "escenario_base", "escenario_bajista", "senales_a_vigilar", "limitaciones"],
}


def evidence_hash(evidence):
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()


def cached_recommendation(evidence):
    key = evidence_hash(evidence)
    return get_db().ai_recommendations.find_one({"evidence_hash": key}, {"_id": 0})


def generate_recommendation(evidence):
    """Gemini redacta escenarios sobre cifras verificables; no calcula el forecast."""
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("Configura GEMINI_API_KEY para generar el análisis.")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    prompt = (
        "Eres analista de riesgos de criptoactivos. Responde en español con escenarios "
        "condicionales y acciones para evaluar, nunca como orden de inversión ni garantía. "
        "Usa exclusivamente este JSON de evidencia; no inventes noticias, probabilidades ni métricas. "
        "Menciona que P10/P90 son intervalos aproximados de un modelo recursivo y que la "
        "fecha de compra/venta es el extremo de la trayectoria central, no una oportunidad cierta. "
        "El sentimiento es una medición VADER de titulares RSS; no equivale a opinión de todo el mercado. "
        "Si el modelo no supera el baseline o la cobertura es baja, prioriza ESPERAR u OBSERVAR. "
        "Datos: " + json.dumps(evidence, ensure_ascii=False, default=str)
    )
    # Conservar vivo el cliente hasta leer la respuesta; una instancia temporal
    # puede cerrarse antes de que termine la petición en algunos entornos.
    with genai.Client(api_key=key) as client:
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA),
        )
        response_text = response.text
    result = json.loads(response_text or "{}")
    missing = set(SCHEMA["required"]) - result.keys()
    if missing or result.get("postura") not in SCHEMA["properties"]["postura"]["enum"]:
        raise ValueError("Gemini devolvió una respuesta incompleta o inválida")
    if evidence.get("beats_baseline") is not True or (evidence.get("coverage_pct") or 0) < 80:
        result["postura"] = "OBSERVAR"
        result["riesgo"] = "ALTO"
        result["limitaciones"] += " Control de calidad: la postura se fijó en OBSERVAR porque el modelo no supera el baseline o la cobertura es insuficiente."
    document = {
        "evidence_hash": evidence_hash(evidence), "coin_id": evidence["coin_id"],
        "horizon_days": evidence["horizon_days"], "forecast_run_id": evidence["forecast_run_id"],
        "generated_at": datetime.now(timezone.utc), "model": model, "evidence": evidence,
        "response": result, "is_latest": True,
    }
    db = get_db()
    db.ai_recommendations.create_index("evidence_hash", unique=True)
    db.ai_recommendations.update_one({"evidence_hash": document["evidence_hash"]}, {"$setOnInsert": document}, upsert=True)
    db.ai_recommendations.update_many(
        {"coin_id": evidence["coin_id"], "evidence_hash": {"$ne": document["evidence_hash"]}, "is_latest": True},
        {"$set": {"is_latest": False}},
    )
    return document


def build_live_evidence(db, asset, horizon_days=30):
    """Combina el modelo vigente con el snapshot y noticias actuales de una moneda."""
    coin_id = asset["coin_id"]
    run = db.forecast_runs.find_one({"coin_id": coin_id, "is_latest": True})
    snapshot = db.market_snapshots.find_one({"coin_id": coin_id}, sort=[("observed_at", -1)])
    metric = db.model_metrics.find_one({"coin_id": coin_id}, sort=[("trained_at", -1)]) or {}
    quality = db.data_quality.find_one({"coin_id": coin_id}, sort=[("checked_at", -1)]) or {}
    if not run or not snapshot or not isinstance(snapshot.get("price"), (int, float)) or snapshot["price"] <= 0:
        return None
    now = datetime.now(timezone.utc)
    base_date = run.get("base_date")
    if not base_date or now - (base_date if base_date.tzinfo else base_date.replace(tzinfo=timezone.utc)) > timedelta(days=10):
        return None
    rows = list(db.forecasts_daily.find(
        {"forecast_run_id": run["forecast_run_id"], "forecast_date": {"$gte": datetime(now.year, now.month, now.day, tzinfo=timezone.utc)}},
        {"_id": 0, "forecast_date": 1, "predicted_price": 1, "p10": 1, "p90": 1}
    ).sort("forecast_date", 1).limit(horizon_days))
    if not rows:
        return None
    target = rows[-1]
    current_price = float(snapshot["price"])
    news = list(db.news.find(
        {"mentioned_coin_ids": coin_id, "published_at": {"$gte": now - timedelta(days=7)}},
        {"_id": 0, "sentiment_compound": 1, "sentiment_label": 1}
    ))
    scores = [float(n["sentiment_compound"]) for n in news if isinstance(n.get("sentiment_compound"), (int, float))]
    forecast_price = float(target["predicted_price"])
    return {
        "analysis_version": 2, "coin_id": coin_id, "symbol": asset.get("symbol"),
        "forecast_run_id": run["forecast_run_id"], "horizon_days": len(rows),
        "snapshot_at": str(snapshot.get("observed_at")), "spot_price_usd": round(current_price, 8),
        "spot_change_24h_pct": snapshot.get("price_change_1d"),
        "forecast_base_date": str(base_date), "forecast_base_price_usd": run.get("base_price"),
        "end_date": str(target["forecast_date"]), "central_price_usd": round(forecast_price, 8),
        "variation_pct": round((forecast_price / current_price - 1) * 100, 3),
        "p10_usd": target.get("p10"), "p90_usd": target.get("p90"),
        "coverage_pct": quality.get("requested_window_coverage_pct"),
        "model_mae": metric.get("mae_log_return"), "baseline_mae": metric.get("baseline_mae_log_return"),
        "beats_baseline": metric.get("beats_zero_return_baseline"),
        "model_trained_at": str(metric.get("trained_at")),
        "news_7d_count": len(news), "news_sentiment_7d_avg": round(sum(scores) / len(scores), 4) if scores else None,
        "news_positive_7d": sum(n.get("sentiment_label") == "POSITIVO" for n in news),
        "news_negative_7d": sum(n.get("sentiment_label") == "NEGATIVO" for n in news),
    }


def generate_all():
    """Genera un dictamen vigente por moneda después de cada actualización ETL."""
    if not os.getenv("GEMINI_API_KEY"):
        return {"status": "skipped_missing_gemini_key", "generated": 0, "cached": 0, "failed": []}
    db = get_db()
    max_assets = max(1, int(os.getenv("AI_MAX_PER_RUN", "20")))
    horizon = max(1, int(os.getenv("AI_HORIZON_DAYS", "30")))
    assets = db.assets.find({"active_top_n": True}, {"_id": 0, "coin_id": 1, "symbol": 1}).sort("rank", 1).limit(max_assets)
    generated, cached, failed = 0, 0, []
    for asset in assets:
        try:
            evidence = build_live_evidence(db, asset, horizon)
            if not evidence:
                continue
            existing = cached_recommendation(evidence)
            if existing:
                cached += 1
                continue
            generate_recommendation(evidence)
            generated += 1
        except Exception as exc:
            failed.append({"coin_id": asset["coin_id"], "error": str(exc)[:300]})
    return {"status": "partial_error" if failed else "ok", "generated": generated, "cached": cached, "failed": failed}
