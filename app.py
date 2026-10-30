"""CryptoAnalytics: tablero gerencial de mercado, pronósticos y escenarios IA."""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse
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
    st.subheader("Noticias y sentimiento")
    news_all = data["news"]
    direct = news_all[news_all["mentioned_coin_ids"].apply(lambda ids: coin_id in ids if isinstance(ids, list) else False)]
    news_view = direct if not direct.empty else news_all
    scope = f"menciones de {asset.get('symbol')}" if not direct.empty else "mercado general · sin menciones directas recientes"
    st.caption(f"Últimos 30 días · {scope}. Sentimiento VADER de titulares y resúmenes RSS; no incluye comentarios de redes sociales.")
    if news_view.empty:
        st.info("No hay noticias recientes almacenadas. Revisa la ejecución del pipeline de noticias.")
    else:
        valid_news = news_view.dropna(subset=["sentiment_compound"])
        pos = int((news_view.sentiment_label == "POSITIVO").sum())
        neg = int((news_view.sentiment_label == "NEGATIVO").sum())
        neutral = int((news_view.sentiment_label == "NEUTRAL").sum())
        a, b, c, d = st.columns(4)
        a.metric("Noticias analizadas", len(news_view))
        b.metric("Tono promedio", f"{valid_news.sentiment_compound.mean():+.2f}" if not valid_news.empty else "Sin dato")
        c.metric("Positivas", pos)
        d.metric("Negativas", neg)
        left, right = st.columns(2, gap="large")
        with left:
            fig = figure("Distribución del tono de noticias", 335, "Noticias")
            fig.add_trace(go.Bar(x=["Positivo", "Neutral", "Negativo"], y=[pos, neutral, neg], marker_color=[GREEN, "#8494aa", RED], text=[pos, neutral, neg], textposition="outside", showlegend=False))
            fig.update_layout(xaxis_title="Clasificación VADER", yaxis_title="Número de noticias")
            st.plotly_chart(fig, width="stretch")
        with right:
            trend = valid_news.assign(day=valid_news.published_at.dt.floor("D")).groupby("day", as_index=False).agg(score=("sentiment_compound", "mean"), count=("title", "count"))
            fig = figure("Evolución diaria del sentimiento", 335, "Puntuación media")
            if not trend.empty:
                fig.add_trace(go.Bar(x=trend.day, y=trend.score, marker_color=[GREEN if v >= 0 else RED for v in trend.score], customdata=trend["count"], hovertemplate="%{x|%d %b}<br>VADER: %{y:+.2f}<br>%{customdata} noticias<extra></extra>", showlegend=False))
            fig.add_hline(y=0, line_dash="dot", line_color="#8694a8")
            fig.update_layout(yaxis_range=[-1, 1])
            st.plotly_chart(fig, width="stretch")
        st.markdown("#### Titulares recientes")
        for _, item in news_view.sort_values("published_at", ascending=False).head(6).iterrows():
            title = re.sub(r"[\[\]<>]", "", str(item.get("title") or "Sin título"))
            url = str(item.get("url") or "")
            host = (urlparse(url).hostname or "").lower()
            badge = {"POSITIVO": "🟢", "NEGATIVO": "🔴"}.get(item.get("sentiment_label"), "⚪")
            if url.startswith("https://") and any(host == d or host.endswith("." + d) for d in ("coindesk.com", "cointelegraph.com", "decrypt.co")):
                st.markdown(f"{badge} [{title}]({url})")
            else:
                st.write(f"{badge} {title}")
            st.caption(f"{item.get('source','Fuente')} · {pd.to_datetime(item['published_at'], utc=True):%d/%m %H:%M UTC} · VADER {float(item['sentiment_compound']):+.2f}" if pd.notna(item.get("sentiment_compound")) else str(item.get("source", "Fuente")))
        latest_news = pd.to_datetime(news_view.published_at.max(), utc=True)
        if (pd.Timestamp.now(tz="UTC") - latest_news).days > 3:
            st.warning("Los últimos titulares almacenados tienen más de tres días; el sentimiento puede estar desactualizado.")
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
    st.subheader("Dictamen y escenarios de IA")
    st.markdown('<div class="lead">El workflow actualiza el dictamen con Gemini después de refrescar mercado y noticias. Se muestra aquí el último análisis guardado para la moneda.</div>', unsafe_allow_html=True)
    recommendation = data["recommendations"].get(coin_id)
    run = data["runs"].get(coin_id)
    base_price = float(snap.get("price") or 0)
    verdict_horizon = int((recommendation or {}).get("horizon_days") or 30)
    future = data["forecasts"][data["forecasts"].coin_id == coin_id].sort_values("forecast_date")
    future = future[future.forecast_date >= pd.Timestamp.now(tz="UTC").normalize()].head(verdict_horizon)
    result = forecast_stats(future, base_price, verdict_horizon)
    if not result:
        st.info("Aún no hay un pronóstico vigente para esta moneda. El dictamen requiere un modelo y un precio actual.")
    else:
        if recommendation:
            answer = recommendation["response"]
            evidence = recommendation["evidence"]
            generated = pd.to_datetime(recommendation["generated_at"], utc=True)
            change = float(evidence.get("variation_pct") or 0)
            direction = "ALCISTA" if change > 2 else "BAJISTA" if change < -2 else "ESTABLE"
            st.caption(f"Gemini {recommendation.get('model','')} · actualizado {generated:%d/%m/%Y %H:%M UTC} · horizonte de {evidence.get('horizon_days',30)} días")
            a, b, c = st.columns(3)
            a.metric("Veredicto de IA", answer["postura"].replace("_", " "))
            b.metric("Riesgo del pronóstico", answer["riesgo"])
            c.metric("Tendencia prevista", direction)
            if (pd.Timestamp.now(tz="UTC") - generated).total_seconds() > 8 * 3600:
                st.warning("El dictamen tiene más de ocho horas. Revisa el último workflow de GitHub Actions.")
            if not recommendation.get("is_latest"):
                st.warning("Este dictamen fue generado antes de la actualización automática; la próxima ejecución del workflow lo renovará.")
            if run and recommendation.get("forecast_run_id") != run.get("forecast_run_id"):
                st.warning("El dictamen corresponde a una versión anterior del pronóstico; espera el próximo workflow.")
            st.markdown("#### Resumen ejecutivo")
            st.write(answer["resumen"])
        else:
            evidence = None
            st.info("Todavía no existe un dictamen de Gemini para esta moneda. Configura GEMINI_API_KEY en GitHub Actions y ejecuta el workflow de actualización.")

        scenario_base = float(evidence.get("spot_price_usd") or base_price) if evidence else base_price
        low = float(evidence.get("p10_usd") or result["p10"]) if evidence else result["p10"]
        central = float(evidence.get("central_price_usd") or result["end_price"]) if evidence else result["end_price"]
        high = float(evidence.get("p90_usd") or result["p90"]) if evidence else result["p90"]
        a, b, c = st.columns(3)
        for column, title, value in [(a, "Escenario bajo · P10", low), (b, "Escenario central", central), (c, "Escenario alto · P90", high)]:
            column.metric(title, money(value), pct((value / scenario_base - 1) * 100))
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            vals = [(v / scenario_base - 1) * 100 for v in (low, central, high)]
            fig = figure("Escenarios al cierre del horizonte · cambio vs. precio actual", 360, "%")
            fig.add_trace(go.Bar(x=["Bajo P10", "Central", "Alto P90"], y=vals, marker_color=[RED, GOLD, GREEN], text=[pct(v) for v in vals], textposition="outside", customdata=[low, central, high], hovertemplate="%{x}<br>%{y:+.1f}%<br>US$ %{customdata:,.4f}<extra></extra>", showlegend=False))
            fig.add_hline(y=0, line_color="#8c9ab0", line_dash="dot")
            fig.update_layout(yaxis_title="Cambio respecto al precio actual (%)")
            st.plotly_chart(fig, width="stretch")
        with right:
            news_for_coin = data["news"][data["news"]["mentioned_coin_ids"].apply(lambda ids: coin_id in ids if isinstance(ids, list) else False)]
            news_week = news_for_coin[news_for_coin.published_at >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)]
            sentiment_counts = [(news_week.sentiment_label == x).sum() for x in ("POSITIVO", "NEUTRAL", "NEGATIVO")]
            fig = figure("Tono de noticias sobre la moneda · 7 días", 360, "Noticias")
            fig.add_trace(go.Bar(x=["Positivo", "Neutral", "Negativo"], y=sentiment_counts, marker_color=[GREEN, "#8494aa", RED], text=sentiment_counts, textposition="outside", showlegend=False))
            fig.update_layout(yaxis_title="Noticias detectadas")
            st.plotly_chart(fig, width="stretch")
            if news_week.empty:
                st.caption("No hay menciones directas recientes; el dictamen debe apoyarse principalmente en precios y calidad del modelo.")
        st.caption("P10–P90 es un rango aproximado del modelo. El sentimiento procede de titulares RSS analizados con VADER; ninguno garantiza el comportamiento futuro.")

        if recommendation:
            st.markdown("#### Tres escenarios para la decisión")
            left, middle, right = st.columns(3, gap="large")
            for column, title, key in [(left, "Si el mercado cae", "escenario_bajista"), (middle, "Caso central", "escenario_base"), (right, "Si el mercado sube", "escenario_alcista")]:
                with column:
                    st.markdown(f"**{title}**")
                    st.write(answer[key])
            st.markdown("#### Señales a vigilar")
            for signal in answer.get("senales_a_vigilar", []):
                st.write("• " + str(signal))
            st.info(answer["limitaciones"])
            with st.expander("Trazabilidad del dictamen"):
                st.json(evidence)
