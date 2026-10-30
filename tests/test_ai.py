import json
import unittest
from unittest.mock import MagicMock, patch

from src.dashboard.ai import generate_recommendation


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


if __name__ == "__main__":
    unittest.main()
