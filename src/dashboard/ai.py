from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import time

from src.db import get_db

MODELOS_A_PROBAR = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 4000

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "recomendacion": {"type": "STRING", "enum": ["COMPRAR", "POSTERGAR_COMPRA", "MANTENER", "VENDER", "POSTERGAR_VENTA"]},
        "riesgo": {"type": "STRING", "enum": ["BAJO", "MODERADO", "ALTO"]},
        "tendencia_general": {"type": "STRING", "enum": ["ALCISTA", "BAJISTA", "LATERAL"]},
        "resumen_ejecutivo": {"type": "STRING"},
        "explicacion_detallada": {"type": "STRING"},
        "mejor_momento_venta": {"type": "STRING"},
        "piso_estimado": {"type": "STRING"},
        "nota_informativa": {"type": "STRING"},
    },
    "required": [
        "recomendacion",
        "riesgo",
        "tendencia_general",
        "resumen_ejecutivo",
        "explicacion_detallada",
        "mejor_momento_venta",
        "piso_estimado",
        "nota_informativa"
    ],
}

def evidence_hash(evidence):
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()

def cached_recommendation(evidence):
    key = evidence_hash(evidence)
    return get_db().ai_recommendations.find_one({"evidence_hash": key}, {"_id": 0})

def registrar_log_agente(
    cantidad_insumos: int,
    modelo_usado: str,
    tiempo_ejecucion_seg: float,
    estado: str,
    tokens_meta: dict = None,
    detalle_error: str = None,
):
    """Registra una traza de auditoría de la ejecución de la IA en la colección 'agent_logs'."""
    try:
        db = get_db()
        log_entry = {
            "fecha_hora": datetime.now(timezone.utc),
            "nombre_agente": "crypto_forecast_agent",
            "modelo_utilizado": modelo_usado,
            "estado": estado,
            "tiempo_ejecucion_segundos": round(tiempo_ejecucion_seg, 3),
            "cantidad_insumos_procesados": cantidad_insumos,
            "configuracion_parametros": {
                "creatividad_temperatura": TEMPERATURE,
                "limite_tokens_salida": MAX_OUTPUT_TOKENS,
            },
            "consumo_tokens": tokens_meta
            or {
                "tokens_entrada_prompt": 0,
                "tokens_salida_respuesta": 0,
                "tokens_totales_consumidos": 0,
            },
            "detalle_error": detalle_error,
        }

        db.agent_logs.insert_one(log_entry)
        print("Registro técnico guardado exitosamente en 'agent_logs'.")
    except Exception as e:
        print(f"Error al guardar el log de auditoría en MongoDB: {e}")


def generate_recommendation(evidence, max_retries=3):
    """Gemini redacta una narrativa con reintentos automáticos ante errores de alta demanda (503)."""
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("Configura GEMINI_API_KEY para generar el análisis.")
    model = os.getenv("GEMINI_MODEL", MODELOS_A_PROBAR[0])
    prompt = (
            "Eres un asesor financiero experto, amigable y excelente comunicador. Tu enfoque principal es que "
            "una persona totalmente nueva en el mundo de las criptomonedas pueda entender el análisis sin dificultad.\n\n"
            "REGLAS DE LENGUAJE Y CLARIDAD:\n"
            "- Utiliza un lenguaje sumamente sencillo, comprensible y práctico, evitando por completo tecnicismos confusos (está prohibido usar palabras bursátiles avanzadas o términos abstractos como 'valles más profundos'; en su lugar, explícalo de forma coloquial como 'momentos de mayor caída' o 'puntos más bajos').\n\n"
            "REGLA DE ORO OBLIGATORIA:\n"
            "- Realiza el análisis asumiendo estrictamente el momento presente del análisis en adelante (hacia el futuro).\n"
            "- Cuando menciones fechas, preséntalas estrictamente en formato de día, mes y año (Ejemplo: 14 de octubre de 2026).\n"
            "- Para los precios, utiliza el formato numérico estándar con separador de miles y dos decimales (Ejemplo: $77,742.51 USD).\n"
            "ESTÁ PROHIBIDO mencionar, referenciar o incluir fechas y eventos pasados. Aunque en los datos se provea información histórica o el punto de partida, tu narrativa debe centrarse de manera exclusiva en la evolución pronosticada hacia adelante a lo largo de los próximos 180 días basándote en los 10 picos altos y 10 picos bajos futuros.\n\n"
            "- Respeta estrictamente fecha_inicio_proyeccion y fecha_fin_proyeccion; no atribuyas predicciones a fechas fuera de ese intervalo.\n"
            "- No inventes precios, fechas ni movimientos que no estén respaldados por los datos suministrados.\n"
            "- No confundas los picos altos o bajos con precio_final_proyectado_usd, que corresponde exclusivamente a fecha_fin_proyeccion.\n"
            "- Una misma fecha no puede tener dos precios centrales proyectados diferentes; ante una inconsistencia, prioriza los datos estructurados y evita afirmaciones contradictorias.\n"
            "- Si hay oscilaciones importantes, no describas la trayectoria como un crecimiento constante.\n"
            "- Las predicciones son estimaciones inciertas: no presentes ganancias ni oportunidades como seguras o garantizadas.\n\n"
            "Reglas para determinar la recomendación y justificarla:\n"
            "- 'COMPRAR': Si el precio actual es bajo y se proyecta un incremento fuerte y sostenido en el futuro cercano.\n"
            "- 'POSTERGAR_COMPRA': Si el precio actual todavía va a bajar más antes de iniciar su tendencia al alza, por lo que conviene esperar al soporte más bajo.\n"
            "- 'MANTENER': Si el precio no fluctúa demasiado, se mantiene estable o en un rango lateral sin cambios drásticos que justifiquen mover la posición.\n"
            "- 'VENDER': Si el precio actual es alto pero se proyecta una caída abrupta y sostenida en el futuro.\n"
            "- 'POSTERGAR_VENTA': Si el precio actual va a tener un bache o bajada temporal, pero en el futuro alcanzará un valor mucho más alto (enfócate en detallar exactamente el día que costará más).\n\n"
            "Estructura de la respuesta:\n"
            "1. 'tendencia_general': Define si la trayectoria global futura es 'ALCISTA', 'BAJISTA' o 'LATERAL'.\n"
            "2. 'recomendacion': Menciona el nombre completo de la moneda analizada y elige estrictamente entre 'COMPRAR', 'POSTERGAR_COMPRA', 'MANTENER', 'VENDER' o 'POSTERGAR_VENTA'.\n"
            "3. 'resumen_ejecutivo': Redacta un párrafo robusto (de al menos 4 líneas) resumiendo el panorama futuro del activo, mencionando el nombre completo de la moneda, su precio actual, hacia dónde se dirige estrictamente en el horizonte pronosticado y mencionando la fecha del cierre estimado.\n"
            "4. 'explicacion_detallada': Redacta una explicación fluida y muy didáctica sobre cómo evolucionará el precio a futuro para principiantes. Menciona el nombre de la moneda y cuenta la historia de la trayectoria de forma orgánica mencionando el precio actual de partida y cómo se moverá el mercado en los próximos días, indicando con fechas exactas futuras qué días alcanzará sus puntos más altos y más bajos (sin tecnicismos), y justificando claramente por qué se eligió la recomendación.\n"
            "5. 'nota_informativa': Redacta un mensaje final sutil, comprensible y amigable que indique que este análisis tiene un enfoque estrictamente informativo, educativo y académico, y que no debe interpretarse como una asesoría financiera especializada.\n\n"
            "Usa exclusivamente la información del JSON de evidencia suministrado:\n"
            + json.dumps(evidence, ensure_ascii=False, default=str)
        )

    start_time = time.time()
    tokens_meta = {}

    attempt = 0
    response_text = None
    last_exception = None

    while attempt < max_retries:
        try:
            with genai.Client(api_key=key) as client:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SCHEMA,
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
                response_text = response.text
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    tokens_meta = {
                        "tokens_entrada_prompt": getattr(usage, "prompt_token_count", 0),
                        "tokens_salida_respuesta": getattr(usage, "candidates_token_count", 0),
                        "tokens_totales_consumidos": getattr(usage, "total_token_count", 0),
                    }
            break
        except Exception as exc:
            last_exception = exc
            error_str = str(exc)
            if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str:
                attempt += 1
                sleep_time = 5 * attempt
                print(f"⚠️ Alerta 503 (Alta demanda) para {evidence['coin_id']}. Reintentando en {sleep_time}s (Intento {attempt}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                raise exc

    if response_text is None:
        raise last_exception

    try:
        result = json.loads(response_text or "{}")
        missing = set(SCHEMA["required"]) - result.keys()
        if missing or result.get("recomendacion") not in SCHEMA["properties"]["recomendacion"]["enum"]:
            raise ValueError("Gemini devolvió una respuesta incompleta o inválida")

        elapsed_time = time.time() - start_time
        document = {
            "evidence_hash": evidence_hash(evidence), "coin_id": evidence["coin_id"],
            "horizon_days": evidence["horizon_days"], "forecast_run_id": evidence.get("forecast_run_id"),
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

        registrar_log_agente(
            cantidad_insumos=1,
            modelo_usado=model,
            tiempo_ejecucion_seg=elapsed_time,
            estado="EXITOSO",
            tokens_meta=tokens_meta,
        )

        return document

    except Exception as exc:
        elapsed_time = time.time() - start_time
        error_msg = str(exc)
        registrar_log_agente(
            cantidad_insumos=1,
            modelo_usado=model,
            tiempo_ejecucion_seg=elapsed_time,
            estado="ERROR",
            detalle_error=error_msg,
        )
        raise exc


def build_live_evidence(db, asset, horizon_days=30):
    """Prepara evidencia solo del ultimo run y de fechas posteriores a hoy (UTC)."""
    coin_id = asset["coin_id"]
    run = db.forecast_runs.find_one({"coin_id": coin_id, "is_latest": True})
    snapshot = db.market_snapshots.find_one({"coin_id": coin_id}, sort=[("observed_at", -1)])
    if not run or not snapshot:
        return None

    try:
        current_price = float(snapshot.get("price"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current_price) or current_price <= 0:
        return None

    run_id = run.get("forecast_run_id")
    if not run_id:
        return None

    now = datetime.now(timezone.utc)
    tomorrow = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)


    cursor = db.forecasts_daily.find(
        {"coin_id": coin_id, "forecast_run_id": run_id, "forecast_date": {"$gte": tomorrow}},
        {"_id": 0, "forecast_date": 1, "predicted_price": 1, "p10": 1, "p90": 1},
    ).sort("forecast_date", 1)

    rows, used_dates = [], set()
    for row in cursor:
        forecast_date = row.get("forecast_date")
        if not isinstance(forecast_date, datetime):
            continue
        try:
            price = float(row.get("predicted_price"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        date_key = forecast_date.date().isoformat()
        if date_key in used_dates:
            continue
        used_dates.add(date_key)
        rows.append({"forecast_date": forecast_date, "predicted_price": price})
        if len(rows) >= horizon_days:
            break

    if not rows:
        return None

    by_price = sorted(rows, key=lambda row: row["predicted_price"])
    lowest_10 = [
        {"fecha": row["forecast_date"].date().isoformat(),
         "precio_usd": round(row["predicted_price"], 2)}
        for row in by_price[:10]
    ]
    highest_10 = [
        {"fecha": row["forecast_date"].date().isoformat(),
         "precio_usd": round(row["predicted_price"], 2)}
        for row in by_price[-10:][::-1]
    ]

    final_row = rows[-1]
    final_price = final_row["predicted_price"]
    variation = round((final_price / current_price - 1) * 100, 2)

    news = list(db.news.find(
        {"mentioned_coin_ids": coin_id, "published_at": {"$gte": now - timedelta(days=7)}},
        {"_id": 0, "sentiment_compound": 1, "sentiment_label": 1},
    ))
    scores = [
        float(item["sentiment_compound"])
        for item in news
        if isinstance(item.get("sentiment_compound"), (int, float))
        and math.isfinite(float(item["sentiment_compound"]))
    ]

    return {
        "coin_id": coin_id,
        "symbol": asset.get("symbol"),
        "forecast_run_id": run_id,
        "horizon_days": len(rows),
        "fecha_analisis_actual": now.date().isoformat(),
        "fecha_inicio_proyeccion": rows[0]["forecast_date"].date().isoformat(),
        "fecha_fin_proyeccion": final_row["forecast_date"].date().isoformat(),
        "spot_price_usd": round(current_price, 2),
        "precio_final_proyectado_usd": round(final_price, 2),
        "variacion_porcentual": variation,
        "top_10_picos_altos": highest_10,
        "top_10_picos_bajos": lowest_10,
        "sentimiento_noticias_7d": round(sum(scores) / len(scores), 2) if scores else 0,
    }


def generate_all():
    """Genera dictámenes actualizados para las monedas activas añadiendo pausa entre peticiones para evitar rate limits."""
    if not os.getenv("GEMINI_API_KEY"):
        return {"status": "skipped_missing_gemini_key", "generated": 0, "cached": 0, "failed": []}
    db = get_db()
    max_assets = max(1, int(os.getenv("AI_MAX_PER_RUN", "20")))
    assets = db.assets.find({"active_top_n": True}, {"_id": 0, "coin_id": 1, "symbol": 1}).sort("rank", 1).limit(max_assets)
    horizon = max(1, int(os.getenv("AI_HORIZON_DAYS", "30")))
    generated, cached, failed = 0, 0, []
    for asset in assets:
        try:
            evidence = build_live_evidence(db, asset, horizon)
            if not evidence:
                continue
            generate_recommendation(evidence)
            generated += 1

            time.sleep(3)
        except Exception as exc:
            failed.append({"coin_id": asset["coin_id"], "error": str(exc)[:300]})
    return {"status": "partial_error" if failed else "ok", "generated": generated, "cached": cached, "failed": failed}
