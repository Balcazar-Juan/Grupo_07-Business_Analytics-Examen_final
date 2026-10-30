from __future__ import annotations

from datetime import datetime, timezone
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
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        "Eres analista de riesgos de criptoactivos. Responde en español con escenarios "
        "condicionales y acciones para evaluar, nunca como orden de inversión ni garantía. "
        "Usa exclusivamente este JSON de evidencia; no inventes noticias, probabilidades ni métricas. "
        "Menciona que P10/P90 son intervalos aproximados de un modelo recursivo y que la "
        "fecha de compra/venta es el extremo de la trayectoria central, no una oportunidad cierta. "
        "Si el modelo no supera el baseline o la cobertura es baja, prioriza ESPERAR u OBSERVAR. "
        "Datos: " + json.dumps(evidence, ensure_ascii=False, default=str)
    )
    response = genai.Client(api_key=key).models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA),
    )
    result = json.loads(response.text or "{}")
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
        "response": result,
    }
    db = get_db()
    db.ai_recommendations.create_index("evidence_hash", unique=True)
    db.ai_recommendations.update_one({"evidence_hash": document["evidence_hash"]}, {"$setOnInsert": document}, upsert=True)
    return document
