"""CryptoAnalytics: tablero gerencial de mercado, pronósticos y escenarios IA."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CryptoAnalytics | Grupo 07", page_icon="₿", layout="wide")
try:
    for name in ("MONGODB_URI", "MONGODB_DB_NAME", "GEMINI_API_KEY", "GEMINI_MODEL"):
        if name in st.secrets and name not in os.environ:
            os.environ[name] = str(st.secrets[name])
except FileNotFoundError:
    pass

from src.dashboard.data import load_real
from src.dashboard.logic import HORIZONS, money, period_stats, forecast_stats
from src.dashboard.ai import cached_recommendation, generate_recommendation

BG, PANEL, GRID = "#0a1020", "#101a2e", "#28344b"
CYAN, GOLD, GREEN, RED = "#51c8e8", "#f6c65e", "#2dd4a4", "#fa7582"
st.markdown("""<style>
.stApp {background:#0a1020;color:#ecf2fa}
.block-container {max-width:1500px;padding-top:1.7rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#0d1729;border-right:1px solid #26354d}
[data-testid="stMetric"] {background:#14223a;border:1px solid #2a3a55;border-radius:14px;padding:16px 18px;min-height:112px}
[data-testid="stMetricLabel"] {color:#9fb2cc}
[data-testid="stMetricValue"] {color:#f6f8fc;font-weight:700}
[data-testid="stMetricValue"] {font-size:clamp(1.15rem,1.7vw,1.65rem)}
[data-testid="stMetricValue"] div {overflow:visible;text-overflow:unset;white-space:normal}
div[data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid #26354d}
button[data-baseweb="tab"] {font-weight:650}
h1,h2,h3 {letter-spacing:-.025em}
.eyebrow {color:#f6c65e;font-size:.81rem;font-weight:750;letter-spacing:.13em;text-transform:uppercase}
.lead {color:#aebcd2;font-size:1rem;margin-bottom:1.1rem}
.panelnote {background:#14243b;border-left:3px solid #51c8e8;padding:12px 16px;border-radius:6px;color:#d8e4f2}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner="Consultando MongoDB Atlas…")
def dashboard_data():
    return load_real()


def symbol(data, coin_id):
    return next((a.get("symbol") or coin_id for a in data["assets"] if a["coin_id"] == coin_id), coin_id)


def pct(value):
    return f"{float(value):+.1f} %" if value is not None and pd.notna(value) else "Sin dato"


def figure(title, height=370, ylabel="USD"):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#ecf2fa")),
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="#b8c8dc"), height=height,
        margin=dict(l=24, r=20, t=58, b=38),
        xaxis=dict(title="Fecha (UTC)", gridcolor=GRID, showline=False),
        yaxis=dict(title=ylabel, gridcolor=GRID, zerolinecolor="#70809b"),
        legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
    )
    return fig


def price_chart(history, forecast, base_date, asset_symbol):
    fig = figure(f"{asset_symbol} · precio observado y trayectoria proyectada", 430)
    recent = history.sort_values("date").tail(90)
    if not recent.empty:
        fig.add_trace(go.Scatter(x=recent.date, y=recent.close, name="Histórico", mode="lines", line=dict(color=CYAN, width=2.6), hovertemplate="%{x|%d %b %Y}<br>US$ %{y:,.4f}<extra>Histórico</extra>"))
    if not forecast.empty:
        fig.add_trace(go.Scatter(x=[base_date, *forecast.forecast_date], y=[float(recent.iloc[-1].close) if not recent.empty else float(forecast.iloc[0].predicted_price), *forecast.predicted_price], name="Proyección central", mode="lines", line=dict(color=GOLD, width=2.8), hovertemplate="%{x|%d %b %Y}<br>US$ %{y:,.4f}<extra>Proyección</extra>"))
        fig.add_vline(x=pd.Timestamp(base_date).timestamp() * 1000, line_dash="dot", line_color="#99a7bc", annotation_text="Inicio del pronóstico", annotation_position="top right")
    return fig


def range_chart(result, base_price):
    fig = figure("Precio al cierre del horizonte · escenarios", 300)
    fig.add_trace(go.Scatter(x=[result["p10"], result["p90"]], y=["Rango estimado"] * 2, mode="lines+markers", line=dict(color="#687b99", width=14), marker=dict(size=10), name="P10–P90", hovertemplate="US$ %{x:,.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[result["end_price"]], y=["Rango estimado"], mode="markers", marker=dict(size=20, color=GOLD, line=dict(color="#fff", width=2)), name="Central", hovertemplate="Central: US$ %{x:,.4f}<extra></extra>"))
    fig.add_vline(x=base_price, line_dash="dash", line_color=CYAN, annotation_text="Precio base", annotation_position="top left")
    fig.update_layout(xaxis_title="Precio en USD", yaxis_title="", showlegend=True)
    return fig


try:
    data = dashboard_data()
except Exception as exc:
    st.error("No se pudieron cargar los datos reales de MongoDB Atlas.")
    st.caption(str(exc))
    st.stop()

assets = data["assets"]
labels = {f"{a.get('symbol','?')} · {a.get('name',a['coin_id'])}": a["coin_id"] for a in assets}
selected = st.sidebar.selectbox("Moneda de análisis", list(labels))
coin_id = labels[selected]
asset = next(a for a in assets if a["coin_id"] == coin_id)
snap = data["snapshots"].get(coin_id, {})
observed = pd.to_datetime(snap.get("observed_at"), utc=True) if snap.get("observed_at") else None
st.sidebar.caption("Datos de MongoDB Atlas · USD")
st.sidebar.caption("Actualizado: " + (observed.strftime("%d/%m/%Y %H:%M UTC") if observed is not None else "sin snapshot"))

st.markdown('<div class="eyebrow">Grupo 07 · Business Analytics</div>', unsafe_allow_html=True)
st.title("CryptoAnalytics")
st.markdown('<div class="lead">Panorama del mercado, proyecciones y escenarios para la toma de decisiones.</div>', unsafe_allow_html=True)
if observed is not None and (pd.Timestamp.now(tz="UTC") - observed).days > 1:
    st.warning("El último snapshot tiene más de un día. Verifica la actualización del pipeline antes de tomar decisiones.")

descriptive, predictive, prescriptive = st.tabs(["Mercado descriptivo", "Pronóstico", "Escenarios con IA"])

with descriptive:
    st.subheader("Pulso del mercado")
    st.caption(f"{asset.get('name')} ({asset.get('symbol')}) · último snapshot de CoinStats en USD")
    a, b, c, d = st.columns(4)
    a.metric("Precio actual", money(snap.get("price")))
    b.metric("Cambio en 24 h", pct(snap.get("price_change_1d")))
    c.metric("Capitalización", money(snap.get("market_cap")))
    d.metric("Volumen 24 h", money(snap.get("volume_24h")))

    prices = data["history"]
    period = st.segmented_control("Periodo histórico", ["7 días", "30 días", "90 días", "1 año"], default="30 días")
    days = {"7 días": 7, "30 días": 30, "90 días": 90, "1 año": 365}[period or "30 días"]
    start = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(int(days), unit="D")
    hist = prices[(prices.coin_id == coin_id) & (prices.date >= start)].sort_values("date")
    stats = period_stats(hist)
    left, right = st.columns([2.25, 1], gap="large")
    with left:
        fig = figure(f"Evolución de {asset.get('symbol')} · cierres diarios", 400)
        if not hist.empty:
            fig.add_trace(go.Scatter(x=hist.date, y=hist.close, mode="lines", fill="tozeroy", fillcolor="rgba(81,200,232,.08)", line=dict(color=CYAN, width=2.6), name="Cierre USD", hovertemplate="%{x|%d %b %Y}<br>US$ %{y:,.4f}<extra></extra>"))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Lectura del periodo")
        if stats:
            st.metric("Rendimiento observado", pct(stats["return_pct"]))
            st.metric("Cierre máximo", money(stats["high"]))
            st.metric("Cierre mínimo", money(stats["low"]))
            st.caption(f"{stats['rows']} cierres entre {stats['start']:%d/%m/%Y} y {stats['end']:%d/%m/%Y}. Son máximos y mínimos de cierre, no intradía.")
            if stats["rows"] < days * .8:
                st.warning("Histórico parcial para el periodo seleccionado.")
        else:
            st.info("No hay cierres suficientes en este periodo.")

    st.subheader("Dónde se mueve el mercado")
    ranking = []
    for item in assets:
        s = data["snapshots"].get(item["coin_id"], {})
        ranking.append({"Moneda": item.get("name"), "Símbolo": item.get("symbol"), "Precio USD": s.get("price"), "24 h %": s.get("price_change_1d"), "7 d %": s.get("price_change_1w"), "Capitalización USD": s.get("market_cap"), "Volumen 24 h USD": s.get("volume_24h"), "Ranking": item.get("rank")})
    rank = pd.DataFrame(ranking)
    rank["24 h %"] = pd.to_numeric(rank["24 h %"], errors="coerce")
    movers = rank.dropna(subset=["24 h %"]).sort_values("24 h %")
    breadth = st.columns(3)
    breadth[0].metric("Activos al alza", int((movers["24 h %"] > 0).sum()))
    breadth[1].metric("Activos a la baja", int((movers["24 h %"] < 0).sum()))
    breadth[2].metric("Mayor cambio 24 h", f"{movers.iloc[-1]['Símbolo']} · {pct(movers.iloc[-1]['24 h %'])}" if not movers.empty else "Sin dato")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        extremes = pd.concat([movers.head(5), movers.tail(5)]).drop_duplicates("Símbolo").sort_values("24 h %")
        fig = figure("Mayores subidas y caídas · 24 h", 405, "%")
        fig.add_trace(go.Bar(x=extremes["24 h %"], y=extremes["Símbolo"], orientation="h", marker_color=[GREEN if v >= 0 else RED for v in extremes["24 h %"]], text=[pct(v) for v in extremes["24 h %"]], textposition="outside", showlegend=False))
        fig.update_layout(xaxis_title="Cambio en 24 h (%)", yaxis_title="", margin=dict(l=30,r=75,t=58,b=38))
        st.plotly_chart(fig, width="stretch")
    with right:
        caps = rank.dropna(subset=["Capitalización USD"]).sort_values("Capitalización USD", ascending=False).head(8)
        fig = figure("Capitalización · principales activos", 405)
        fig.add_trace(go.Bar(x=caps["Capitalización USD"], y=caps["Símbolo"], orientation="h", marker_color=CYAN, showlegend=False, hovertemplate="%{y}: US$ %{x:,.0f}<extra></extra>"))
        fig.update_layout(xaxis_title="USD", yaxis_title="", yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Comparación de rendimiento")
    defaults = [cid for cid in ["bitcoin", "ethereum", "solana", "binance-coin"] if cid in set(prices.coin_id)]
    chosen = st.multiselect("Activos para comparar", options=[a["coin_id"] for a in assets], default=defaults, format_func=lambda cid: symbol(data, cid))
    comparison = prices[(prices.coin_id.isin(chosen)) & (prices.date >= start)].sort_values(["coin_id", "date"]).copy()
    fig = figure("Rendimiento desde el inicio del periodo · misma escala para todas las monedas", 410, "%")
    palette = [CYAN, GOLD, GREEN, "#b99aff", RED, "#79a9ff"]
    for i, (cid, group) in enumerate(comparison.groupby("coin_id")):
        first = float(group.iloc[0].close)
        if first > 0:
            fig.add_trace(go.Scatter(x=group.date, y=(group.close / first - 1) * 100, mode="lines", name=symbol(data, cid), line=dict(color=palette[i % len(palette)], width=2.4)))
    fig.add_hline(y=0, line_color="#78869c", line_dash="dot")
    st.plotly_chart(fig, width="stretch")
    st.caption("El 0 % es el primer cierre disponible de cada moneda en el periodo. Así se comparan tendencias sin mezclar precios de distinto tamaño.")
    st.dataframe(rank.sort_values("Ranking"), width="stretch", hide_index=True, column_config={
        "Precio USD": st.column_config.NumberColumn("Precio USD", format="$%.4f"),
        "24 h %": st.column_config.NumberColumn("24 h %", format="%+.2f%%"),
        "7 d %": st.column_config.NumberColumn("7 d %", format="%+.2f%%"),
        "Capitalización USD": st.column_config.NumberColumn("Capitalización USD", format="$%.0f"),
        "Volumen 24 h USD": st.column_config.NumberColumn("Volumen 24 h USD", format="$%.0f"),
    })

with predictive:
    st.subheader("Pronóstico de precios")
    st.markdown('<div class="lead">Selecciona el horizonte y lee la trayectoria junto al último histórico. El rango final resume la incertidumbre.</div>', unsafe_allow_html=True)
    horizon_label = st.select_slider("Horizonte del análisis", options=list(HORIZONS), value="3 meses")
    horizon = HORIZONS[horizon_label]
    individual, comparison_tab = st.tabs(["Una moneda", "Comparar monedas"])
    forecasts = data["forecasts"]
    with individual:
        run = data["runs"].get(coin_id)
        future = forecasts[forecasts.coin_id == coin_id].sort_values("forecast_date").head(horizon)
        result = forecast_stats(future, float((run or {}).get("base_price") or 0), horizon)
        if not run or not result:
            st.info("Esta moneda aún no tiene un pronóstico vigente. Selecciona otra moneda con histórico suficiente.")
        else:
            base_price = float(run["base_price"])
            base_date = pd.to_datetime(run["base_date"], utc=True)
            metric = data["metrics"].get(coin_id, {})
            if metric.get("beats_zero_return_baseline") is False:
                st.warning("Confiabilidad limitada: XGBoost no mejoró el pronóstico simple de precio sin cambio en su validación. Usa los escenarios como exploración.")
            if result["days_available"] < horizon:
                st.warning(f"Solo hay {result['days_available']} días de proyección para este horizonte.")
            a, b, c, d = st.columns(4)
            a.metric("Precio base", money(base_price))
            b.metric("Precio al final", money(result["end_price"]))
            c.metric("Cambio proyectado", pct(result["variation_pct"]))
            d.metric("Dirección general", result["trend"])
            hist = data["history"][data["history"].coin_id == coin_id]
            st.plotly_chart(price_chart(hist, future, base_date, asset.get("symbol")), width="stretch")
            left, right = st.columns([1.1, 1], gap="large")
            with left:
                st.plotly_chart(range_chart(result, base_price), width="stretch")
                st.caption("P10 y P90 son límites aproximados al final del horizonte. El punto amarillo es la trayectoria central; la línea azul indica el precio base.")
            with right:
                st.markdown("#### Fechas destacadas de la trayectoria central")
                st.metric("Menor cierre proyectado", f"{pd.Timestamp(result['buy']['forecast_date']):%d/%m/%Y} · {money(result['buy']['predicted_price'])}")
                st.metric("Mayor cierre proyectado", f"{pd.Timestamp(result['sell']['forecast_date']):%d/%m/%Y} · {money(result['sell']['predicted_price'])}")
                st.caption("Son extremos del modelo dentro del horizonte seleccionado. No indican el mejor momento real para operar.")
            st.markdown("#### Hitos de la proyección")
            milestones = [v for v in [15, 30, 60, 90, 180, 270, 365] if v <= len(future) and v <= horizon]
            rows = []
            for day in milestones:
                row = future.iloc[day - 1]
                rows.append({"Día": day, "Fecha": row.forecast_date.date(), "Precio central USD": row.predicted_price, "Cambio vs. base %": round((row.predicted_price / base_price - 1) * 100, 2), "Límite bajo P10 USD": row.p10, "Límite alto P90 USD": row.p90})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
                "Precio central USD": st.column_config.NumberColumn("Precio central USD", format="$%.2f"),
                "Cambio vs. base %": st.column_config.NumberColumn("Cambio vs. base %", format="%+.2f%%"),
                "Límite bajo P10 USD": st.column_config.NumberColumn("Límite bajo P10 USD", format="$%.2f"),
                "Límite alto P90 USD": st.column_config.NumberColumn("Límite alto P90 USD", format="$%.2f"),
            })
            st.caption(f"Pronóstico XGBoost generado el {pd.to_datetime(run['generated_at'], utc=True):%d/%m/%Y %H:%M UTC}. Cuanto más lejano el día, mayor incertidumbre.")
    with comparison_tab:
        available = [a["coin_id"] for a in assets if a["coin_id"] in data["runs"]]
        defaults = [cid for cid in ["bitcoin", "ethereum", "solana", "binance-coin"] if cid in available]
        chosen = st.multiselect("Monedas con pronóstico", available, default=defaults, format_func=lambda cid: symbol(data, cid), key="compare_predictive")
        fig = figure("Cambio proyectado desde el precio base", 460, "%")
        summary = []
        for i, cid in enumerate(chosen):
            run = data["runs"][cid]
            group = forecasts[forecasts.coin_id == cid].sort_values("forecast_date").head(horizon)
            info = forecast_stats(group, float(run["base_price"]), horizon)
            if info:
                change = (group.predicted_price / float(run["base_price"]) - 1) * 100
                fig.add_trace(go.Scatter(x=group.forecast_date, y=change, mode="lines", name=symbol(data, cid), line=dict(color=palette[i % len(palette)], width=2.4)))
                summary.append({"Moneda": symbol(data, cid), "Cambio al final %": round(info["variation_pct"], 2), "Precio final USD": info["end_price"], "Tendencia": info["trend"], "Cobertura del horizonte": f"{info['days_available']}/{horizon} días"})
        fig.add_hline(y=0, line_color="#8996aa", line_dash="dot")
        st.plotly_chart(fig, width="stretch")
        st.caption("Cada línea empieza en el precio base de su moneda. Así se compara el cambio porcentual sin poner BTC y activos de bajo precio en el mismo eje USD.")
        if summary:
            final = pd.DataFrame(summary).sort_values("Cambio al final %", ascending=False)
            bar = figure("Cambio estimado al final del horizonte", 350, "%")
            bar.add_trace(go.Bar(x=final["Moneda"], y=final["Cambio al final %"], marker_color=[GREEN if x >= 0 else RED for x in final["Cambio al final %"]], text=[pct(x) for x in final["Cambio al final %"]], textposition="outside", showlegend=False))
            bar.update_layout(xaxis_title="Moneda", yaxis_title="Cambio estimado (%)")
            st.plotly_chart(bar, width="stretch")
            st.dataframe(final, width="stretch", hide_index=True, column_config={
                "Cambio al final %": st.column_config.NumberColumn("Cambio al final %", format="%+.2f%%"),
                "Precio final USD": st.column_config.NumberColumn("Precio final USD", format="$%.4f"),
            })

with prescriptive:
    st.subheader("Escenarios para decidir qué vigilar")
    st.markdown('<div class="lead">Gemini resume un pronóstico real y sus límites. La recomendación es una postura para evaluar, no una orden de compra o venta.</div>', unsafe_allow_html=True)
    run = data["runs"].get(coin_id)
    future = data["forecasts"][data["forecasts"].coin_id == coin_id].sort_values("forecast_date").head(horizon)
    result = forecast_stats(future, float((run or {}).get("base_price") or 0), horizon)
    if not run or not result:
        st.info("Selecciona una moneda con pronóstico vigente para ver escenarios y generar el análisis de IA.")
    else:
        base_price = float(run["base_price"])
        q = data["quality"].get(coin_id, {})
        m = data["metrics"].get(coin_id, {})
        evidence = {"coin_id": coin_id, "symbol": asset.get("symbol"), "horizon_days": horizon, "forecast_run_id": run["forecast_run_id"], "base_date": str(run.get("base_date")), "base_price_usd": round(base_price, 8), "end_date": str(result["end"]), "central_price_usd": round(result["end_price"], 8), "variation_pct": round(result["variation_pct"], 3), "p10_usd": round(result["p10"], 8), "p90_usd": round(result["p90"], 8), "coverage_pct": q.get("requested_window_coverage_pct"), "model_mae": m.get("mae_log_return"), "baseline_mae": m.get("baseline_mae_log_return"), "beats_baseline": m.get("beats_zero_return_baseline"), "model_trained_at": str(m.get("trained_at"))}
        st.caption(f"{asset.get('symbol')} · {horizon_label} · precio base {money(base_price)} · fecha objetivo {pd.Timestamp(result['end']):%d/%m/%Y}")
        a, b, c = st.columns(3)
        for column, title, value in [(a, "Escenario bajo · P10", result["p10"]), (b, "Escenario central", result["end_price"]), (c, "Escenario alto · P90", result["p90"])]:
            column.metric(title, money(value), pct((value / base_price - 1) * 100))
        st.caption("P10–P90 es un rango aproximado calculado con errores históricos del modelo; no es una probabilidad garantizada.")
        if m.get("beats_zero_return_baseline") is False:
            st.warning("La validación del modelo es débil frente al baseline. La postura de IA se limita a OBSERVAR.")
        try:
            recommendation = cached_recommendation(evidence)
        except Exception as exc:
            recommendation = None
            st.warning(f"No se pudo consultar el análisis almacenado: {exc}")
        if st.button("Generar o actualizar análisis con Gemini", type="primary", disabled=not os.getenv("GEMINI_API_KEY")):
            try:
                with st.spinner("Gemini analiza los escenarios…"):
                    recommendation = generate_recommendation(evidence)
            except Exception as exc:
                st.error(f"No se pudo generar el análisis de Gemini: {exc}")
        if not os.getenv("GEMINI_API_KEY"):
            st.info("Configura GEMINI_API_KEY en los secretos de Streamlit para activar el análisis de IA.")
        if recommendation:
            answer = recommendation["response"]
            st.divider()
            a, b = st.columns([1, 1])
            a.metric("Postura sugerida", answer["postura"].replace("_", " "))
            b.metric("Riesgo del pronóstico", answer["riesgo"])
            st.markdown("#### Resumen ejecutivo")
            st.write(answer["resumen"])
            left, middle, right = st.columns(3, gap="large")
            for column, title, key in [(left, "Si cae", "escenario_bajista"), (middle, "Caso central", "escenario_base"), (right, "Si sube", "escenario_alcista")]:
                with column:
                    st.markdown(f"#### {title}")
                    st.write(answer[key])
            st.markdown("#### Señales a vigilar")
            for signal in answer["senales_a_vigilar"]:
                st.write("• " + signal)
            st.info(answer["limitaciones"])
            st.caption(f"Gemini {recommendation['model']} · análisis guardado el {pd.to_datetime(recommendation['generated_at'], utc=True):%d/%m/%Y %H:%M UTC}")
        with st.expander("Datos utilizados por la IA"):
            st.json(evidence)
