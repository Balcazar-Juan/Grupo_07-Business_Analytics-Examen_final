import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.dashboard.ai import build_live_evidence, generate_all, generate_recommendation


class Response:
    def __init__(self, client):
        self.client = client

    @property
    def text(self):
        if self.client.closed:
            raise RuntimeError("client already closed")
        return json.dumps({
            "postura": "OBSERVAR", "riesgo": "ALTO", "resumen": "Revisar señales.",
            "escenario_alcista": "Sube.", "escenario_base": "Se mantiene.",
            "escenario_bajista": "Cae.", "senales_a_vigilar": ["Volumen"],
            "limitaciones": "Pronóstico incierto.",
        })


class FakeClient:
    closed = False

    def __init__(self, **kwargs):
        self.models = MagicMock()
        self.models.generate_content.return_value = Response(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class GeminiTests(unittest.TestCase):
    def test_response_is_read_before_client_closes(self):
        evidence = {"coin_id": "bitcoin", "forecast_run_id": "run1", "horizon_days": 30, "beats_baseline": False, "coverage_pct": 99}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}), patch("google.genai.Client", FakeClient), patch("src.dashboard.ai.get_db", return_value=MagicMock()):
            result = generate_recommendation(evidence)
        self.assertEqual(result["response"]["postura"], "OBSERVAR")
        self.assertEqual(result["response"]["riesgo"], "ALTO")

    def test_live_evidence_joins_current_price_forecast_and_news(self):
        now = datetime.now(timezone.utc)
        db = MagicMock()
        db.forecast_runs.find_one.return_value = {
            "forecast_run_id": "run1", "base_date": now - timedelta(days=1), "base_price": 100,
        }
        db.market_snapshots.find_one.return_value = {"price": 110, "observed_at": now}
        db.model_metrics.find_one.return_value = {"beats_zero_return_baseline": True}
        db.data_quality.find_one.return_value = {"requested_window_coverage_pct": 95}
        db.forecasts_daily.find.return_value.sort.return_value.limit.return_value = [
            {"forecast_date": now + timedelta(days=30), "predicted_price": 121, "p10": 90, "p90": 130}
        ]
        db.news.find.return_value = [
            {"sentiment_compound": 0.5, "sentiment_label": "POSITIVO"},
            {"sentiment_compound": -0.3, "sentiment_label": "NEGATIVO"},
        ]
        evidence = build_live_evidence(db, {"coin_id": "bitcoin", "symbol": "BTC"})
        self.assertEqual(evidence["variation_pct"], 10.0)
        self.assertEqual(evidence["news_7d_count"], 2)
        self.assertEqual(evidence["news_sentiment_7d_avg"], 0.1)

    def test_refresh_generates_saved_verdict_for_available_asset(self):
        db = MagicMock()
        db.assets.find.return_value.sort.return_value.limit.return_value = [{"coin_id": "bitcoin"}]
        evidence = {"coin_id": "bitcoin", "forecast_run_id": "run1", "horizon_days": 30}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}), \
             patch("src.dashboard.ai.get_db", return_value=db), \
             patch("src.dashboard.ai.build_live_evidence", return_value=evidence), \
             patch("src.dashboard.ai.cached_recommendation", return_value=None), \
             patch("src.dashboard.ai.generate_recommendation") as generate:
            result = generate_all()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["generated"], 1)
        generate.assert_called_once_with(evidence)


if __name__ == "__main__":
    unittest.main()
