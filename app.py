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
        if {"p10", "p90"}.issubset(forecast.columns):
            forecast_band_x = [base_date, *forecast.forecast_date]
            forecast_base = float(recent.iloc[-1].close) if not recent.empty else float(forecast.iloc[0].predicted_price)
            fig.add_trace(go.Scatter(x=forecast_band_x, y=[forecast_base, *forecast.p90], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=forecast_band_x, y=[forecast_base, *forecast.p10], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(246,198,94,.14)", name="Rango P10–P90", hovertemplate="P10–P90<extra></extra>"))
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

ignored_global_symbols = {"USDC", "USDT", "STETH", "XMR", "WSTETH", "USDS", "BSC-USD", "WBT", "HYPE"}
assets = [
    a for a in assets 
    if a.get("symbol") not in ignored_global_symbols 
    and a["coin_id"] not in {"usdc", "usdt", "steth", "xmr", "wsteth", "usds", "bsc-usd", "wbt", "hyperliquid"}
]
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
    st.markdown('<div class="lead">Selecciona el horizonte y lee la trayectoria proyectada junto al último precio real del mercado.</div>', unsafe_allow_html=True)
    st.info("**Aviso académico:** Las estimaciones mostradas provienen de modelos de Machine Learning (XGBoost) con fines puramente analíticos y educativos. Deben tomarse con precaución y no constituyen asesoría financiera.", icon="ℹ️")
    
    horizon_label = st.select_slider("Horizonte del análisis (Máximo 1 año)", options=["30 días", "90 días", "180 días", "1 año"], value="90 días")
    HORIZONS_MAP = {"30 días": 30, "90 días": 90, "180 días": 180, "1 año": 365}
    horizon = HORIZONS_MAP[horizon_label]
    
    individual, comparison_tab = st.tabs(["Una moneda", "Comparar monedas"])
    forecasts = data["forecasts"]
    
    with individual:
        run = data["runs"].get(coin_id)
        future = forecasts[forecasts.coin_id == coin_id].sort_values("forecast_date").head(horizon)
        result = forecast_stats(future, float((run or {}).get("base_price") or 0), horizon)
        
        if not run or not result:
            st.info("Esta moneda aún no tiene un pronóstico vigente. Selecciona otra moneda con histórico suficiente.")
        else:
            hist_coin = data["history"][data["history"].coin_id == coin_id]
            base_price = float(hist_coin["close"].iloc[-1]) if not hist_coin.empty else float(run["base_price"])
            base_date = pd.to_datetime(run["base_date"], utc=True)
            metric = data["metrics"].get(coin_id, {})
            
            if result["days_available"] < horizon:
                st.warning(f"Solo hay {result['days_available']} días de proyección para este horizonte.")
            
            a, b, c, d = st.columns(4)
            a.metric("Precio actual (Mercado)", money(base_price))
            b.metric("Precio al final del periodo", money(result["end_price"]))
            c.metric("Cambio proyectado", pct(result["variation_pct"]))
            d.metric("Dirección general al final", result["trend"])
      
            st.plotly_chart(price_chart(hist_coin, future, base_date, asset.get("symbol")), width="stretch")
      
            st.markdown("---")
            st.markdown("### Puntos Clave para Operar")
            st.markdown("Identificación automática de los días estimados con menor y mayor valor dentro del periodo seleccionado.")
            
            col_buy, col_sell = st.columns(2, gap="medium")
            
            with col_buy:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(14, 209, 143, 0.1); border: 1px solid #0ed18f; padding: 20px; border-radius: 10px;">
                        <h4 style="color: #0ed18f; margin: 0 0 10px 0;">🟢 Mejor día para comprar</h4>
                        <p style="font-size: 22px; font-weight: bold; margin: 0 0 5px 0;">{money(result['buy']['predicted_price'])}</p>
                        <p style="color: #a0aec0; margin: 0; font-size: 14px;">Fecha estimada: <b>{pd.Timestamp(result['buy']['forecast_date']):%d/%m/%Y}</b></p>
                        <hr style="border-color: rgba(14, 209, 143, 0.2); margin: 10px 0;">
                        <p style="font-size: 12px; color: #a0aec0; margin: 0;">Día futuro en el que se estima que el precio será el más bajo.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with col_sell:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(234, 179, 8, 0.1); border: 1px solid #EAB308; padding: 20px; border-radius: 10px;">
                        <h4 style="color: #EAB308; margin: 0 0 10px 0;">🟡 Mejor día para vender</h4>
                        <p style="font-size: 22px; font-weight: bold; margin: 0 0 5px 0;">{money(result['sell']['predicted_price'])}</p>
                        <p style="color: #a0aec0; margin: 0; font-size: 14px;">Fecha estimada: <b>{pd.Timestamp(result['sell']['forecast_date']):%d/%m/%Y}</b></p>
                        <hr style="border-color: rgba(234, 179, 8, 0.2); margin: 10px 0;">
                        <p style="font-size: 12px; color: #a0aec0; margin: 0;">Día futuro en el que se estima que el precio alcanzará su punto más alto.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
        with st.expander("Métricas de validación del modelo"):
            st.caption("Calculadas con validación walk-forward sobre precios históricos para medir la confiabilidad del modelo de predicción.")
            
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("RMSE relativo", f"{float(metric.get('rmse_price_pct', 0)):.1f}%")
            m2.metric("MSE relativo", f"{float(metric.get('mse_price_pct2', 0)):.2f}%²" if metric.get("mse_price_pct2") is not None else "Sin dato")
            m3.metric("MAPE", f"{float(metric.get('mape_price_pct', 0)):.1f}%")
            m4.metric("MAE relativo", f"{float(metric.get('mae_price_pct', 0)):.1f}%")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.caption("Representa el margen de error global en porcentaje frente al valor real del activo.")
            with d2:
                st.caption("Eleva al cuadrado los errores para penalizar y detectar desviaciones grandes del modelo.")
            with d3:
                st.caption("Muestra el porcentaje promedio de equivocación directa entre el precio estimado y el real.")
            with d4:
                st.caption("Mide la diferencia promedio absoluta en porcentaje entre la predicción y el mercado.")
            
        st.caption(f"Pronóstico XGBoost generado el {pd.to_datetime(run['generated_at'], utc=True):%d/%m/%Y %H:%M UTC}. Cuanto más lejano el día, mayor incertidumbre.")

    with comparison_tab:
            ignored_symbols = {"USDC", "USDT", "STETH", "XMR", "WSTETH", "USDS", "BSC-USD", "WBT"}
            available_assets = [
                a for a in assets 
                if a["coin_id"] in data["runs"] and a.get("symbol") not in ignored_symbols
            ]
            
            symbol_options = [a.get("symbol") for a in available_assets if a.get("symbol")]
            default_symbols = [s for s in ["BTC", "ETH", "SOL", "BNB"] if s in symbol_options]
            
            st.markdown("#### Activos para comparar")
            selected_symbols = st.multiselect(
                "Selecciona las criptomonedas para el análisis comparativo de pronóstico:",
                options=symbol_options,
                default=default_symbols,
                key="multiselect_forecast_comparison"
            )
            
            symbol_to_coin_id = {a.get("symbol"): a["coin_id"] for a in available_assets}
            chosen = [symbol_to_coin_id[sym] for sym in selected_symbols if sym in symbol_to_coin_id]
            
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
                st.markdown("#### Ranking de mejores criptomonedas según rendimiento proyectado")
                st.dataframe(final, width="stretch", hide_index=True, column_config={
                    "Cambio al final %": st.column_config.NumberColumn("Cambio al final %", format="%+.2f%%"),
                    "Precio final USD": st.column_config.NumberColumn("Precio final USD", format="$%.4f"),
                })

with prescriptive:
    st.subheader("Dictamen y Perspectiva del Asesor IA")
    st.markdown(
        '<div class="lead">Análisis predictivo actualizado de forma automática para acompañar tus decisiones de inversión de manera sencilla y clara.</div>', 
        unsafe_allow_html=True
    )
    nota_default = "Este análisis tiene un enfoque estrictamente informativo, educativo y académico, y no constituye asesoría financiera especializada."
    st.markdown(
        f"""<div style="background-color: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; padding: 15px; border-radius: 8px; color: #93c5fd; font-size: 0.9rem; margin-bottom: 20px;">
            <b>Aviso académico y de investigación:</b> {nota_default}
        </div>""",
        unsafe_allow_html=True
    )

    recommendation = None
    recs_store = data.get("ai_recommendations") if "ai_recommendations" in data else data.get("recommendations")
    
    if isinstance(recs_store, dict):
        recommendation = recs_store.get(coin_id)
    elif hasattr(recs_store, "empty") and not recs_store.empty:
        coin_recs = recs_store[recs_store["coin_id"] == coin_id]
        if not coin_recs.empty:
            coin_recs = coin_recs.sort_values("generated_at", ascending=False)
            recommendation = coin_recs.iloc[0].to_dict()

    run = data["runs"].get(coin_id) if isinstance(data.get("runs"), dict) else None
    base_price = float(snap.get("price") or 0)
    verdict_horizon = int((recommendation or {}).get("horizon_days") or 30)
    
    if recommendation is None:
        st.info("Todavía no existe un dictamen registrado para esta moneda.")
    else:
        answer = recommendation.get("response", recommendation)
        evidence = recommendation.get("evidence", {})
        
        generated_raw = recommendation.get("generated_at")
        if generated_raw:
            generated = pd.to_datetime(generated_raw, utc=True)
            fecha_str = f"{generated:%d/%m/%Y a las %H:%M UTC}"
        else:
            fecha_str = "Reciente"
        veredicto_ia = str(answer.get("recomendacion", "MANTENER")).replace("_", " ").upper()
        riesgo_ia = str(answer.get("riesgo", "MODERADO")).upper()
        tendencia_raw = str(answer.get("tendencia_general", "LATERAL")).upper()

        tendencia_ia = "EN ALZA 📈" if "ALCISTA" in tendencia_raw else "EN BAJA 📉" if "BAJISTA" in tendencia_raw else "ESTABLE / LATERAL ⚖️"
        
        resumen_ejecutivo = answer.get("resumen_ejecutivo", "")
        explicacion_detallada = answer.get("explicacion_detallada", "")
        mejor_momento_venta = answer.get("mejor_momento_venta", "Por definir")
        piso_estimado = answer.get("piso_estimado", "Por definir")
        
        fecha_piso = "Próximamente"
        if evidence and "top_10_picos_bajos" in evidence and evidence["top_10_picos_bajos"]:
            bajos = sorted(evidence["top_10_picos_bajos"], key=lambda x: float(x.get("precio_usd", 0)))
            if bajos:
                fecha_piso = bajos[0].get("fecha", "Próximamente")

        st.caption(f"🤖 Análisis generado por IA · Actualizado el {fecha_str}")

        col1, col2, col3 = st.columns(3)
        
        rec_color = "#10b981" if "COMPRAR" in veredicto_ia else "#ef4444" if "VENDER" in veredicto_ia else "#f59e0b"
        with col1:
            st.markdown(
                f"""<div style="background-color: #1e293b; border-left: 5px solid {rec_color}; padding: 15px; border-radius: 8px;">
                    <small style="color: #94a3b8; font-weight: bold;">RECOMENDACIÓN SUGERIDA</small>
                    <h3 style="color: #f8fafc; margin: 5px 0 0 0;">{veredicto_ia}</h3>
                </div>""", 
                unsafe_allow_html=True
            )
        
        riesgo_color = "#10b981" if "BAJO" in riesgo_ia else "#f59e0b" if "MODERADO" in riesgo_ia else "#ef4444"
        with col2:
            st.markdown(
                f"""<div style="background-color: #1e293b; border-left: 5px solid {riesgo_color}; padding: 15px; border-radius: 8px;">
                    <small style="color: #94a3b8; font-weight: bold;">NIVEL DE RIESGO DE INVERSIÓN</small>
                    <h3 style="color: #f8fafc; margin: 5px 0 0 0;">{riesgo_ia}</h3>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""<div style="background-color: #1e293b; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px;">
                    <small style="color: #94a3b8; font-weight: bold;">TENDENCIA GENERAL</small>
                    <h3 style="color: #f8fafc; margin: 5px 0 0 0; font-size: 1.2rem;">{tendencia_ia}</h3>
                </div>""", 
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Resumen Ejecutivo")
        
        col_img1, col_txt1 = st.columns([0.7, 4], gap="medium")
        
        with col_img1:
            st.image("https://cdn-icons-png.flaticon.com/512/4559/4559130.png", use_container_width=True)
            
        with col_txt1:
            st.markdown(
                f"""<div style='font-size: 1.1rem; line-height: 1.6; color: #f1f5f9;'>
                    {resumen_ejecutivo}
                </div>""", 
                unsafe_allow_html=True
            )

        st.markdown("### ¿Cómo evolucionará el precio?")
        col_txt2, col_img2 = st.columns([3, 1], gap="medium")
        with col_txt2:
            texto_explicacion = explicacion_detallada
            for keyword in ["COMPRAR", "VENDER", "MANTENER", "POSTERGAR_VENTA", "POSTERGAR_COMPRA"]:
                if keyword in texto_explicacion:
                    texto_explicacion = texto_explicacion.replace(
                        f"la recomendación de {keyword}", 
                        f"**<span style='color: #38bdf8; font-size: 1.1em;'>LA RECOMENDACIÓN CLAVE ES {keyword}</span>**"
                    )
            st.markdown(f"<div style='font-size: 1.05rem; line-height: 1.6;'>{texto_explicacion}</div>", unsafe_allow_html=True)
            
        with col_img2:
            st.image("https://images.squarespace-cdn.com/content/v1/61e98da4007e4616fc0daad3/59dcf9d6-d897-4b75-863d-9e6e44525760/AdobeStock_1188413489.png", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Puntos Clave a Tener en Cuenta")
        kp1, kp2 = st.columns(2, gap="medium")
        
        with kp1:
            st.markdown(
                f"""<div style="background-color: #0f172a; border: 1px solid #334155; padding: 20px; border-radius: 10px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.9rem; font-weight: bold; margin-bottom: 8px;">PRECIO MÍNIMO ESPERADO (SOPORTE)</div>
                    <div style="color: #38bdf8; font-size: 1.4rem; font-weight: bold;">{piso_estimado}</div>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 5px;">📅 Fecha estimada: {fecha_piso}</div>
                </div>""",
                unsafe_allow_html=True
            )
            
        with kp2:
            st.markdown(
                f"""<div style="background-color: #0f172a; border: 1px solid #334155; padding: 20px; border-radius: 10px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.9rem; font-weight: bold; margin-bottom: 8px;">MOMENTO IDEAL DE VENTA</div>
                    <div style="color: #34d399; font-size: 1.05rem; font-weight: bold; word-break: break-word;">{mejor_momento_venta}</div>
                </div>""",
                unsafe_allow_html=True
            )
