import unittest
import numpy as np
import pandas as pd

from src.dashboard.logic import forecast_stats, normalized, period_stats
from src.ml.features import build


class CalculationTests(unittest.TestCase):
    def test_period_uses_chronological_endpoints_and_range(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"], utc=True), "close": [120, 100, 80]})
        result = period_stats(frame)
        self.assertEqual(result["low"], 80)
        self.assertEqual(result["high"] - result["low"], 40)
        self.assertAlmostEqual(result["return_pct"], 20)

    def test_forecast_extremes_are_within_selected_horizon(self):
        frame = pd.DataFrame({"forecast_date": pd.date_range("2026-01-02", periods=3, tz="UTC"), "predicted_price": [90, 110, 150], "p10": [80, 90, 100], "p90": [100, 130, 200]})
        result = forecast_stats(frame, 100, 2)
        self.assertAlmostEqual(result["variation_pct"], 10)
        self.assertEqual(result["sell"]["predicted_price"], 110)

    def test_normalization_keeps_assets_separate(self):
        frame = pd.DataFrame({"coin_id": ["btc", "btc", "eth", "eth"], "date": pd.date_range("2026-01-01", periods=2, tz="UTC").tolist() * 2, "close": [100, 110, 10, 9]})
        result = normalized(frame, "date", "close")
        self.assertEqual(result["change_pct"].round().tolist(), [0, 10, 0, -10])

    def test_missing_calendar_day_breaks_features(self):
        dates = pd.date_range("2026-01-01", periods=50, tz="UTC").delete(48)
        frame = pd.DataFrame({"date": dates, "close": np.linspace(100, 150, len(dates)), "volume": 1000})
        result = build(frame, target=False)
        self.assertNotIn(pd.Timestamp("2026-02-19", tz="UTC"), set(result["date"]))


if __name__ == "__main__":
    unittest.main()
