from __future__ import annotations

import numpy as np
import pandas as pd


HORIZONS = {"15 días": 15, "1 mes": 30, "2 meses": 60, "3 meses": 90, "6 meses": 180, "9 meses": 270, "1 año": 365}


def money(value):
    if value is None or not np.isfinite(float(value)):
        return "Sin dato"
    value = float(value)
    if abs(value) >= 1e12:
        return f"US$ {value / 1e12:,.2f} billones"
    if abs(value) >= 1e9:
        return f"US$ {value / 1e9:,.2f} mil M"
    if abs(value) >= 1e6:
        return f"US$ {value / 1e6:,.2f} M"
    return f"US$ {value:,.6f}" if abs(value) < 1 else f"US$ {value:,.2f}"


def period_stats(frame: pd.DataFrame):
    frame = frame.sort_values("date")
    if len(frame) < 2:
        return None
    first, last = frame.iloc[0], frame.iloc[-1]
    return {
        "first": float(first["close"]), "last": float(last["close"]),
        "low": float(frame["close"].min()), "high": float(frame["close"].max()),
        "return_pct": (float(last["close"]) / float(first["close"]) - 1) * 100,
        "start": first["date"], "end": last["date"], "rows": len(frame),
    }


def forecast_stats(frame: pd.DataFrame, base_price: float, days: int):
    frame = frame.sort_values("forecast_date").head(days)
    if frame.empty or base_price <= 0:
        return None
    last = frame.iloc[-1]
    low = frame.loc[frame["predicted_price"].idxmin()]
    high = frame.loc[frame["predicted_price"].idxmax()]
    change = (float(last["predicted_price"]) / base_price - 1) * 100
    trend = "Alcista" if change > 2 else "Bajista" if change < -2 else "Estable (±2 %)"
    return {"end": last["forecast_date"], "end_price": float(last["predicted_price"]),
            "variation_pct": change, "trend": trend, "buy": low, "sell": high,
            "p10": float(last["p10"]), "p90": float(last["p90"]), "days_available": len(frame)}


def normalized(frame, date_col, price_col):
    out = frame.copy().sort_values(["coin_id", date_col])
    out["change_pct"] = out.groupby("coin_id")[price_col].transform(lambda x: (x / x.iloc[0] - 1) * 100)
    return out
