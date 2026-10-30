from __future__ import annotations
import numpy as np
import pandas as pd

# Variables tecnicas pensadas para poder proyectar recursivamente sin requerir
# datos futuros de noticias ni valores absolutos fuera del rango historico.
FEATURES = [
    "return_1d", "return_3d", "return_7d", "return_14d",
    "ma_ratio_7", "ma_ratio_14", "ma_ratio_30",
    "ema_ratio_7", "ema_ratio_21",
    "volatility_7", "volatility_14", "volatility_30",
    "momentum_7", "momentum_14", "rsi_14",
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_7",
    "volume_change_1d", "volume_z_30",
]


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build(prices: pd.DataFrame, target: bool = True) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    df.loc[df["close"] <= 0, "close"] = np.nan
    # Los indicadores de N días requieren N días de calendario consecutivos.
    # Un día ausente corta la ventana; nunca se interpreta como retorno de 1 día.
    df = df.set_index("date").asfreq("D").reset_index()

    ret1 = df["close"].pct_change(fill_method=None)
    for n in (1, 3, 7, 14):
        df[f"return_{n}d"] = df["close"].pct_change(n, fill_method=None)

    for n in (7, 14, 30):
        ma = df["close"].rolling(n).mean()
        df[f"ma_ratio_{n}"] = df["close"] / ma - 1

    ema7 = df["close"].ewm(span=7, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_ratio_7"] = df["close"] / ema7 - 1
    df["ema_ratio_21"] = df["close"] / ema21 - 1

    for n in (7, 14, 30):
        df[f"volatility_{n}"] = ret1.rolling(n).std()

    df["momentum_7"] = df["close"] / df["close"].shift(7) - 1
    df["momentum_14"] = df["close"] / df["close"].shift(14) - 1
    df["rsi_14"] = rsi(df["close"])

    for n in (1, 2, 3, 7):
        df[f"lag_return_{n}"] = ret1.shift(n)

    vol = df["volume"].replace(0, np.nan)
    df["volume_change_1d"] = vol.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_mean = vol.rolling(30).mean()
    vol_std = vol.rolling(30).std().replace(0, np.nan)
    df["volume_z_30"] = ((vol - vol_mean) / vol_std).fillna(0.0)

    # Retorno logaritmico del dia siguiente: estable para proyeccion recursiva.
    if target:
        df["target_log_return_next"] = np.log(df["close"].shift(-1) / df["close"])

    needed = FEATURES + (["target_log_return_next"] if target else [])
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=needed).reset_index(drop=True)
