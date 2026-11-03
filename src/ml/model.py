from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import math
import numpy as np
import pandas as pd
from bson.binary import Binary
from pymongo import UpdateOne
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor, Booster, DMatrix
from src.config import settings
from src.db import get_db
from src.ml.features import FEATURES, build

STABLE = {"USDT", "USDC", "DAI", "FDUSD", "USDE", "USDS", "PYUSD", "TUSD", "FRAX"}
DIRECT_HORIZONS = (1, 7, 15, 30, 90, 180, 365)


def is_stable(asset):
    return bool(asset.get("is_stablecoin")) or str(asset.get("symbol", "")).upper() in STABLE


def price_df(db, coin_id):
    rows = list(db.market_daily.find(
        {"coin_id": coin_id},
        {"_id": 0, "date": 1, "close": 1, "volume": 1, "source": 1}
    ).sort("date", 1))
    if not rows:
        return pd.DataFrame(columns=["date", "close", "volume"]), "market_daily"
    return pd.DataFrame(rows), "market_daily"


def persist_features(coin_id: str, feat: pd.DataFrame):
    if feat.empty:
        return 0
    db = get_db()
    asset = db.assets.find_one({"coin_id": coin_id}, {"_id": 0, "symbol": 1}) or {}
    now = datetime.now(timezone.utc)
    ops = []
    for _, row in feat.iterrows():
        doc = {
            "coin_id": coin_id,
            "symbol": asset.get("symbol"),
            "date": pd.Timestamp(row["date"]).to_pydatetime(),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0) or 0),
            **{f: float(row[f]) for f in FEATURES},
            "updated_at": now,
        }
        if "target_log_return_next" in row and pd.notna(row["target_log_return_next"]):
            doc["target_log_return_next"] = float(row["target_log_return_next"])
        ops.append(UpdateOne({"coin_id": coin_id, "date": doc["date"]}, {"$set": doc}, upsert=True))
    if ops:
        db.features_daily.bulk_write(ops, ordered=False)
    return len(ops)


DEFAULT_PARAMS = {
        "n_estimators": 900, "max_depth": 3, "learning_rate": 0.03,
        "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85,
        "reg_lambda": 4.0, "reg_alpha": 0.10, "gamma": 0.05,
}


def _new_model(params=None):
    config = {**DEFAULT_PARAMS, **(params or {})}
    return XGBRegressor(
        objective="reg:squarederror",
        **config,
        random_state=42,
        n_jobs=2,
    )


def _select_params(feat: pd.DataFrame):
    """Selecciona hiperparámetros en un corte temporal, sin mirar el futuro."""
    if len(feat) < 600:
        return dict(DEFAULT_PARAMS)
    split = int(len(feat) * 0.8)
    train, valid = feat.iloc[:split], feat.iloc[split:]
    candidates = [
        DEFAULT_PARAMS,
        {**DEFAULT_PARAMS, "max_depth": 2, "min_child_weight": 8, "reg_lambda": 8.0},
        {**DEFAULT_PARAMS, "max_depth": 4, "min_child_weight": 3, "learning_rate": 0.02, "n_estimators": 1200},
    ]
    scores = []
    for params in candidates:
        model = _new_model(params)
        model.fit(train[FEATURES], train["target_log_return_next"], verbose=False)
        pred = model.predict(valid[FEATURES])
        scores.append((mean_absolute_error(valid["target_log_return_next"], pred), params))
    return dict(min(scores, key=lambda item: item[0])[1])


def _evaluate_walk_forward(feat: pd.DataFrame, params=None):
    # Hasta 4 folds; en series cortas usa 3. Nunca mezcla futuro en entrenamiento.
    n_splits = 4 if len(feat) >= 800 else 3
    # El target de la última fila de train usa el cierre del primer día de test.
    # Se purga una observación en cada corte para evitar esa fuga temporal.
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)
    rows = []
    residuals = []
    actual_all, pred_all = [], []
    actual_price_all, pred_price_all = [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(feat), start=1):
        train = feat.iloc[train_idx]
        test = feat.iloc[test_idx]
        model = _new_model(params)
        model.fit(train[FEATURES], train["target_log_return_next"])
        pred = model.predict(test[FEATURES])
        actual = test["target_log_return_next"].to_numpy()
        current_prices = test["close"].to_numpy(dtype=float)
        actual_price = current_prices * np.exp(actual)
        pred_price = current_prices * np.exp(pred)
        residual = actual - pred
        residuals.extend(residual.tolist())
        actual_all.extend(actual.tolist())
        pred_all.extend(pred.tolist())
        actual_price_all.extend(actual_price.tolist())
        pred_price_all.extend(pred_price.tolist())

        rows.append({
            "fold": fold,
            "train_start": pd.Timestamp(train.iloc[0]["date"]).to_pydatetime(),
            "train_end": pd.Timestamp(train.iloc[-1]["date"]).to_pydatetime(),
            "test_start": pd.Timestamp(test.iloc[0]["date"]).to_pydatetime(),
            "test_end": pd.Timestamp(test.iloc[-1]["date"]).to_pydatetime(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "mae_log_return": float(mean_absolute_error(actual, pred)),
            "rmse_log_return": float(np.sqrt(mean_squared_error(actual, pred))),
            "mae_price_usd": float(mean_absolute_error(actual_price, pred_price)),
            "mse_price_usd": float(mean_squared_error(actual_price, pred_price)),
            "rmse_price_usd": float(np.sqrt(mean_squared_error(actual_price, pred_price))),
            "mape_price_pct": float(np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100),
            "direction_accuracy_pct": float((np.sign(actual) == np.sign(pred)).mean() * 100),
        })

    actual_all = np.asarray(actual_all)
    pred_all = np.asarray(pred_all)
    actual_price_all = np.asarray(actual_price_all)
    pred_price_all = np.asarray(pred_price_all)
    baseline = np.zeros_like(actual_all)
    return {
        "folds": rows,
        "mae_log_return": float(mean_absolute_error(actual_all, pred_all)),
        "rmse_log_return": float(np.sqrt(mean_squared_error(actual_all, pred_all))),
        "mae_price_usd": float(mean_absolute_error(actual_price_all, pred_price_all)),
        "mse_price_usd": float(mean_squared_error(actual_price_all, pred_price_all)),
        "rmse_price_usd": float(np.sqrt(mean_squared_error(actual_price_all, pred_price_all))),
        "mape_price_pct": float(np.mean(np.abs((actual_price_all - pred_price_all) / actual_price_all)) * 100),
        "direction_accuracy_pct": float((np.sign(actual_all) == np.sign(pred_all)).mean() * 100),
        "baseline_mae_log_return": float(mean_absolute_error(actual_all, baseline)),
        "beats_zero_return_baseline": bool(mean_absolute_error(actual_all, pred_all) < mean_absolute_error(actual_all, baseline)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
    }


def _train_direct_models(db, coin_id, asset, prices, selected_params, source, trained_at):
    """Entrena un modelo independiente para cada horizonte y evita deriva recursiva."""
    base = build(prices, target=False)
    direct_metrics = {}
    for horizon in DIRECT_HORIZONS:
        frame = base.copy()
        target_col = f"target_log_return_{horizon}d"
        frame[target_col] = np.log(frame["close"].shift(-horizon) / frame["close"])
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [target_col])
        if len(frame) < settings.model_min_rows:
            continue
        split = max(int(len(frame) * 0.8), len(frame) - 180)
        split = min(max(split, 1), len(frame) - 1)
        train, valid = frame.iloc[:split], frame.iloc[split:]
        model = _new_model(selected_params)
        model.fit(train[FEATURES], train[target_col], verbose=False)
        pred = model.predict(valid[FEATURES])
        actual = valid[target_col].to_numpy()
        current = valid["close"].to_numpy(dtype=float)
        actual_price = current * np.exp(actual)
        pred_price = current * np.exp(pred)
        residual_std = float(np.std(actual - pred, ddof=1)) if len(actual) > 1 else 0.0
        raw = bytes(model.get_booster().save_raw(raw_format="json"))
        q01, q99 = np.quantile(frame[target_col], [0.01, 0.99])
        db.models_direct.update_one(
            {"coin_id": coin_id, "horizon_days": horizon},
            {"$set": {
                "coin_id": coin_id, "symbol": asset.get("symbol"), "horizon_days": horizon,
                "model_raw_json": Binary(raw), "feature_columns": FEATURES,
                "model_type": "XGBRegressorDirect", "trained_at": trained_at,
                "model_version": trained_at.strftime("%Y%m%dT%H%M%SZ"),
                "history_source": source, "training_rows": int(len(frame)),
                "residual_std": residual_std,
                "return_clip_low": max(float(q01), -abs(settings.forecast_return_clip)),
                "return_clip_high": min(float(q99), abs(settings.forecast_return_clip)),
                "mae_price_usd": float(mean_absolute_error(actual_price, pred_price)),
                "rmse_price_usd": float(np.sqrt(mean_squared_error(actual_price, pred_price))),
                "mape_price_pct": float(np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100),
            }}, upsert=True,
        )
        direct_metrics[str(horizon)] = {"rows": int(len(frame)), "rmse_price_usd": float(np.sqrt(mean_squared_error(actual_price, pred_price)))}
    return direct_metrics


def train_one(coin_id):
    db = get_db()
    asset = db.assets.find_one({"coin_id": coin_id}) or {}
    if not asset:
        return {"coin_id": coin_id, "status": "asset_not_found"}
    if is_stable(asset):
        return {"coin_id": coin_id, "status": "skipped_stablecoin"}

    prices, source = price_df(db, coin_id)
    if not prices.empty and (datetime.now(timezone.utc) - pd.to_datetime(prices.iloc[-1]["date"], utc=True).to_pydatetime()).days > 3:
        return {"coin_id": coin_id, "status": "stale_history", "last_date": str(prices.iloc[-1]["date"])}
    feat = build(prices, target=True)
    if len(feat) < settings.model_min_rows:
        return {"coin_id": coin_id, "status": "insufficient_history", "rows": len(feat)}

    persist_features(coin_id, feat)
    selected_params = _select_params(feat)
    evaluation = _evaluate_walk_forward(feat, selected_params)

    model = _new_model(selected_params)
    model.fit(feat[FEATURES], feat["target_log_return_next"])
    trained_at = datetime.now(timezone.utc)

    target = feat["target_log_return_next"].to_numpy()
    q01, q99 = np.quantile(target, [0.01, 0.99])
    # Doble salvaguarda para evitar trayectorias absurdas por un unico dia extremo.
    clip_low = max(float(q01), -abs(settings.forecast_return_clip))
    clip_high = min(float(q99), abs(settings.forecast_return_clip))

    feature_importance = {
        feature: float(value)
        for feature, value in zip(FEATURES, model.feature_importances_)
    }

    model_version = trained_at.strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "coin_id": coin_id,
        "symbol": asset.get("symbol"),
        "trained_at": trained_at,
        "model_version": model_version,
        "history_source": source,
        "training_start": pd.Timestamp(feat.iloc[0]["date"]).to_pydatetime(),
        "training_end": pd.Timestamp(feat.iloc[-1]["date"]).to_pydatetime(),
        "training_rows": int(len(feat)),
        "target": "next_day_log_return",
        "hyperparameters": selected_params,
        **evaluation,
        "feature_importance": feature_importance,
        "forecast_return_clip_low": clip_low,
        "forecast_return_clip_high": clip_high,
    }

    raw = bytes(model.get_booster().save_raw(raw_format="json"))
    db.models.update_one({"coin_id": coin_id}, {"$set": {
        "coin_id": coin_id,
        "symbol": asset.get("symbol"),
        "history_source": source,
        "feature_columns": FEATURES,
        "model_type": "XGBRegressor",
        "target": "next_day_log_return",
        "model_raw_json": Binary(raw),
        "trained_at": trained_at,
        "training_start": pd.Timestamp(feat.iloc[0]["date"]).to_pydatetime(),
        "training_end": pd.Timestamp(feat.iloc[-1]["date"]).to_pydatetime(),
        "model_version": model_version,
        "hyperparameters": selected_params,
        "residual_std": evaluation["residual_std"],
        "return_clip_low": clip_low,
        "return_clip_high": clip_high,
    }}, upsert=True)
    db.model_metrics.insert_one(metrics)
    direct_metrics = _train_direct_models(db, coin_id, asset, prices, selected_params, source, trained_at)
    db.model_metrics.update_one({"_id": db.model_metrics.find_one({"coin_id": coin_id, "trained_at": trained_at}, {"_id": 1})["_id"]}, {"$set": {"direct_horizon_metrics": direct_metrics}})
    return {"status": "trained", **metrics}


def train_all():
    db = get_db()
    ids = [x["coin_id"] for x in db.assets.find({"active_top_n": True}, {"coin_id": 1}).sort("rank", 1)]
    return [train_one(x) for x in ids]


def _load_booster(model_doc):
    booster = Booster()
    booster.load_model(bytearray(model_doc["model_raw_json"]))
    return booster


def _direct_forecast_docs(db, coin_id, asset, prices, source, days, model_version, generated_at, run_id):
    direct = list(db.models_direct.find({"coin_id": coin_id}, {"_id": 0}).sort("horizon_days", 1))
    direct = [x for x in direct if int(x.get("horizon_days", 0)) <= days]
    if not direct:
        return []
    feat = build(prices, target=False)
    if feat.empty:
        return []
    X = pd.DataFrame([feat.iloc[-1][FEATURES].to_dict()])[FEATURES]
    base_price = float(prices.iloc[-1]["close"])
    last_date = pd.Timestamp(prices.iloc[-1]["date"])
    anchors = [(0, math.log(base_price), 0.0)]
    for doc in direct:
        booster = _load_booster(doc)
        pred_return = float(booster.predict(DMatrix(X, feature_names=FEATURES))[0])
        pred_return = float(np.clip(pred_return, doc.get("return_clip_low", -0.25), doc.get("return_clip_high", 0.25)))
        uncertainty = 1.2815515655446004 * float(doc.get("residual_std") or 0.0)
        anchors.append((int(doc["horizon_days"]), math.log(base_price) + pred_return, uncertainty))
    anchors.sort(key=lambda x: x[0])
    hs = np.array([x[0] for x in anchors], dtype=float)
    logs = np.array([x[1] for x in anchors], dtype=float)
    errs = np.array([x[2] for x in anchors], dtype=float)
    docs = []
    previous = base_price
    for day in range(1, days + 1):
        log_price = float(np.interp(day, hs, logs))
        uncertainty = float(np.interp(day, hs, errs))
        price = math.exp(log_price)
        lower = math.exp(log_price - uncertainty)
        upper = math.exp(log_price + uncertainty)
        forecast_date = last_date + pd.Timedelta(days=day)
        docs.append({
            "forecast_run_id": run_id, "coin_id": coin_id, "symbol": asset.get("symbol"),
            "generated_at": generated_at, "forecast_date": forecast_date.to_pydatetime(),
            "horizon_day": day, "predicted_price": price, "p10": lower, "p50": price, "p90": upper,
            "predicted_log_return": math.log(price / previous), "model": "XGBoostDirectHorizons",
            "model_version": model_version, "history_source": source, "is_latest": True,
        })
        previous = price
    return docs


def forecast_one(coin_id: str, days: int | None = None):
    db = get_db()
    days = int(days or settings.forecast_days)
    model_doc = db.models.find_one({"coin_id": coin_id})
    if not model_doc:
        return {"coin_id": coin_id, "status": "no_model"}

    asset = db.assets.find_one({"coin_id": coin_id}) or {}
    prices, source = price_df(db, coin_id)
    if prices.empty:
        return {"coin_id": coin_id, "status": "no_market_daily"}
    if (datetime.now(timezone.utc) - pd.to_datetime(prices.iloc[-1]["date"], utc=True).to_pydatetime()).days > 3:
        db.forecast_runs.update_many({"coin_id": coin_id, "is_latest": True}, {"$set": {"is_latest": False}})
        db.forecasts_daily.update_many({"coin_id": coin_id, "is_latest": True}, {"$set": {"is_latest": False}})
        return {"coin_id": coin_id, "status": "stale_history", "last_date": str(prices.iloc[-1]["date"])}

    prices = prices[["date", "close", "volume"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True)
    prices = prices.sort_values("date").drop_duplicates("date", keep="last")

    booster = _load_booster(model_doc)
    generated_at = datetime.now(timezone.utc)
    model_version = model_doc.get("model_version") or generated_at.strftime("%Y%m%dT%H%M%SZ")
    run_key = f"{coin_id}|{model_version}|{generated_at.isoformat()}|{days}"
    run_id = hashlib.sha256(run_key.encode()).hexdigest()[:24]

    direct_docs = _direct_forecast_docs(db, coin_id, asset, prices, source, days, model_version, generated_at, run_id)
    if direct_docs:
        db.forecasts_daily.update_many({"coin_id": coin_id, "is_latest": True}, {"$set": {"is_latest": False}})
        db.forecast_runs.update_many({"coin_id": coin_id, "is_latest": True}, {"$set": {"is_latest": False}})
        db.forecasts_daily.bulk_write([
            UpdateOne({"forecast_run_id": run_id, "coin_id": coin_id, "forecast_date": doc["forecast_date"]}, {"$set": doc}, upsert=True)
            for doc in direct_docs
        ], ordered=False)
        db.forecast_runs.update_one({"forecast_run_id": run_id}, {"$set": {
            "forecast_run_id": run_id, "coin_id": coin_id, "symbol": asset.get("symbol"),
            "generated_at": generated_at, "forecast_days": days,
            "forecast_start": (pd.Timestamp(prices.iloc[-1]["date"]) + pd.Timedelta(days=1)).to_pydatetime(),
            "forecast_end": (pd.Timestamp(prices.iloc[-1]["date"]) + pd.Timedelta(days=days)).to_pydatetime(),
            "base_date": pd.Timestamp(prices.iloc[-1]["date"]).to_pydatetime(),
            "base_price": float(prices.iloc[-1]["close"]), "model_version": model_version,
            "interval": "direct_horizon_models_interpolated_p10_p90", "is_latest": True,
        }}, upsert=True)
        return {"coin_id": coin_id, "status": "forecasted", "forecast_run_id": run_id, "days": days,
                "base_price": float(prices.iloc[-1]["close"]), "final_predicted_price": direct_docs[-1]["predicted_price"],
                "forecast_method": "direct_horizons"}

    residual_std = float(model_doc.get("residual_std") or 0.0)
    clip_low = float(model_doc.get("return_clip_low", -abs(settings.forecast_return_clip)))
    clip_high = float(model_doc.get("return_clip_high", abs(settings.forecast_return_clip)))

    # Volumen futuro no se pronostica: se mantiene una mediana reciente para que
    # sus features no introduzcan ceros artificiales.
    nonzero_vol = pd.to_numeric(prices["volume"], errors="coerce").replace(0, np.nan).dropna()
    future_volume = float(nonzero_vol.tail(30).median()) if not nonzero_vol.empty else 0.0

    last_real_date = pd.Timestamp(prices.iloc[-1]["date"])
    last_real_price = float(prices.iloc[-1]["close"])
    path = prices.copy()
    ops = []
    cumulative_pred_log_return = 0.0

    for horizon in range(1, days + 1):
        feat = build(path, target=False)
        if feat.empty:
            return {"coin_id": coin_id, "status": "insufficient_features_during_forecast", "horizon": horizon}
        last = feat.iloc[-1]
        X = pd.DataFrame([last[FEATURES].to_dict()])[FEATURES]
        pred_log_return = float(booster.predict(DMatrix(X, feature_names=FEATURES))[0])
        pred_log_return = float(np.clip(pred_log_return, clip_low, clip_high))

        current_price = float(path.iloc[-1]["close"])
        next_price = max(1e-12, current_price * math.exp(pred_log_return))
        next_date = last_real_date + pd.Timedelta(days=horizon)
        cumulative_pred_log_return += pred_log_return

        # Intervalo aproximado P10-P90 basado en error walk-forward acumulado.
        uncertainty = 1.2815515655446004 * residual_std * math.sqrt(horizon)
        central_log_price = math.log(last_real_price) + cumulative_pred_log_return
        lower = math.exp(central_log_price - uncertainty)
        upper = math.exp(central_log_price + uncertainty)

        doc = {
            "forecast_run_id": run_id,
            "coin_id": coin_id,
            "symbol": asset.get("symbol"),
            "generated_at": generated_at,
            "forecast_date": next_date.to_pydatetime(),
            "horizon_day": horizon,
            "predicted_price": float(next_price),
            "p10": float(lower),
            "p50": float(next_price),
            "p90": float(upper),
            "predicted_log_return": pred_log_return,
            "model": "XGBoost",
            "model_version": model_version,
            "history_source": source,
            "training_start": model_doc.get("training_start"),
            "training_end": model_doc.get("training_end"),
            "is_latest": True,
        }
        ops.append(UpdateOne(
            {"forecast_run_id": run_id, "coin_id": coin_id, "forecast_date": doc["forecast_date"]},
            {"$set": doc}, upsert=True
        ))
        path = pd.concat([path, pd.DataFrame([{
            "date": next_date,
            "close": next_price,
            "volume": future_volume,
        }])], ignore_index=True)

    # Marca la corrida anterior como historica y facilita a Power BI tomar solo la vigente.
    db.forecasts_daily.update_many({"coin_id": coin_id, "is_latest": True}, {"$set": {"is_latest": False}})
    db.forecast_runs.update_many({"coin_id": coin_id, "is_latest": True}, {"$set": {"is_latest": False}})
    if ops:
        db.forecasts_daily.bulk_write(ops, ordered=False)

    db.forecast_runs.update_one({"forecast_run_id": run_id}, {"$set": {
        "forecast_run_id": run_id,
        "coin_id": coin_id,
        "symbol": asset.get("symbol"),
        "generated_at": generated_at,
        "forecast_days": days,
        "forecast_start": (last_real_date + pd.Timedelta(days=1)).to_pydatetime(),
        "forecast_end": (last_real_date + pd.Timedelta(days=days)).to_pydatetime(),
        "base_date": last_real_date.to_pydatetime(),
        "base_price": last_real_price,
        "model_version": model_version,
        "residual_std": residual_std,
        "interval": "approx_p10_p90_from_walk_forward_residual_std",
        "is_latest": True,
    }}, upsert=True)

    return {
        "coin_id": coin_id,
        "status": "forecasted",
        "forecast_run_id": run_id,
        "days": days,
        "base_price": last_real_price,
        "final_predicted_price": float(path.iloc[-1]["close"]),
    }


def forecast_all(days: int | None = None):
    """Genera forecast solo para los activos vigentes del Top N.

    Evita arrastrar modelos antiguos de monedas que ya no forman parte del universo
    activo. Los stablecoins pueden aparecer con estado ``no_model`` porque se omiten
    deliberadamente durante el entrenamiento.
    """
    db = get_db()
    ids = [
        x["coin_id"]
        for x in db.assets.find({"active_top_n": True}, {"coin_id": 1}).sort("rank", 1)
    ]
    return [forecast_one(x, days=days) for x in ids]
