"""CryptoAnalytics: descriptivo, predictivo y escenarios con Gemini."""
from __future__ import annotations

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CryptoAnalytics | Grupo 07", page_icon="₿", layout="wide")
try:
    for secret_name in ("MONGODB_URI", "MONGODB_DB_NAME", "GEMINI_API_KEY", "GEMINI_MODEL"):
        if secret_name in st.secrets and secret_name not in os.environ:
            os.environ[secret_name] = str(st.secrets[secret_name])
except FileNotFoundError:
    pass

from src.dashboard.data import load_real, load_demo
from src.dashboard.logic import HORIZONS, money, period_stats, forecast_stats, normalized
from src.dashboard.ai import cached_recommendation, generate_recommendation

st.markdown("""<style>
 .stApp {background:#0b1120;color:#e7eef9}
 [data-testid="stMetric"] {background:#162238;border:1px solid #2c3f5a;border-radius:12px;padding:14px}
 h1,h2,h3 {color:#f7fafc}
 .subtle {color:#a9bad1;font-size:.92rem}
 .badge {background:#14394b;color:#63d9cc;border-radius:7px;padding:4px 10px;font-size:.85rem}
 </style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner="Cargando datos de MongoDB…")
def dashboard_data(demo):
    return load_demo() if demo else load_real()


def line_chart(frame, x, y, title, shade=False):
    fig = go.Figure()
    if shade and {"p10", "p90"}.issubset(frame.columns):
        fig.add_trace(go.Scatter(x=frame[x], y=frame["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=frame[x], y=frame["p10"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(48,187,169,.15)", name="P10–P90 aprox.", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame[x], y=frame[y], mode="lines", line=dict(color="#36d8bf", width=2.7), name="Precio"))
    fig.update_layout(title=title, template="plotly_dark", paper_bgcolor="#111b2c", plot_bgcolor="#111b2c", height=400, margin=dict(l=20,r=20,t=55,b=25), xaxis_title="Fecha (UTC)", yaxis_title="USD")
    return fig


def asset_label(asset):
    return f"{asset.get('symbol', '?')} · {asset.get('name', asset['coin_id'])}"


st.title("₿ CryptoAnalytics")
st.caption("Grupo 07 · Mercado, pronósticos XGBoost y escenarios asistidos por Gemini · Valores en USD")

demo_requested = st.sidebar.toggle("Datos de demostración", value=False, help="Series ficticias para probar la interfaz.")
try:
    data = dashboard_data(demo_requested)
except Exception as exc:
    st.warning(f"No hay datos reales disponibles: {exc}")
    data = dashboard_data(True)

if data["demo"]:
    st.error("MODO DEMOSTRACIÓN · Todos los precios y pronósticos mostrados son ficticios. La IA está deshabilitada.")
else:
    st.success("Fuente: MongoDB Atlas · CoinStats / Binance / Mobula según disponibilidad")

if st.sidebar.button("Actualizar lectura"):
    dashboard_data.clear()
    st.rerun()

assets = data["assets"]
labels = {asset_label(a): a["coin_id"] for a in assets}
selected_label = st.sidebar.selectbox("Criptomoneda principal", list(labels))
coin_id = labels[selected_label]
asset = next(a for a in assets if a["coin_id"] == coin_id)
snap = data["snapshots"].get(coin_id, {})
observed = snap.get("observed_at")
st.sidebar.caption("Último snapshot: " + (pd.Timestamp(observed).strftime("%Y-%m-%d %H:%M UTC") if observed else "sin dato"))

tab_descriptive, tab_predictive, tab_ai, tab_quality = st.tabs(["📊 Descriptivo", "📈 Predictivo", "✨ Prescriptivo IA", "🧪 Calidad de datos"])

with tab_descriptive:
    st.header("Mercado actual")
    period = st.selectbox("Periodo histórico", ["24 horas", "7 días", "30 días", "90 días", "1 año"], index=2, key="period")
    period_days = {"24 horas":1,"7 días":7,"30 días":30,"90 días":90,"1 año":365}[period]
    period_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(int(period_days), unit="D")
    history = data["history"]
    coin_hist = history[(history.coin_id == coin_id) & (history.date >= period_start)]
    stats = period_stats(coin_hist)
    cols = st.columns(4)
    cols[0].metric("Precio último snapshot", money(snap.get("price")))
    cols[1].metric("Variación 24 h", f"{snap['price_change_1d']:+.2f} %" if snap.get("price_change_1d") is not None else "Sin dato")
    cols[2].metric("Capitalización", money(snap.get("market_cap")))
    cols[3].metric("Volumen 24 h", money(snap.get("volume_24h")))
    if stats:
        expected = period_days + 1
        if len(coin_hist) < min(expected * .8, expected - 1):
            st.warning(f"Cobertura parcial: {len(coin_hist)} observaciones en el periodo solicitado. Interpreta los extremos con cautela.")
        st.plotly_chart(line_chart(coin_hist,"date","close",f"Precio histórico · {asset['symbol']}"), width="stretch")
        a,b,c,d = st.columns(4)
        a.metric("Mínimo del periodo", money(stats["low"]))
        b.metric("Máximo del periodo", money(stats["high"]))
        c.metric("Rango", money(stats["high"]-stats["low"]))
        d.metric("Rendimiento", f"{stats['return_pct']:+.2f} %")
        st.caption(f"Primer cierre: {stats['start']:%Y-%m-%d} · Último cierre: {stats['end']:%Y-%m-%d}. Extremos de cierres diarios; no equivalen al máximo/mínimo intradía.")
    else:
        st.info("Aún no hay suficientes cierres históricos en el periodo elegido.")

    st.subheader("Comparación de activos")
    defaults = [x for x in labels if x.split(" · ")[0] in ("BTC","ETH","SOL","BNB")]
    compare_labels = st.multiselect("Monedas", list(labels), default=defaults, key="descriptive_compare")
    compare_ids = [labels[x] for x in compare_labels]
    compare = history[(history.coin_id.isin(compare_ids)) & (history.date >= period_start)].copy()
    if not compare.empty:
        normalized_view = st.toggle("Comparar rendimiento desde el inicio (%)", value=True)
        plot = normalized(compare,"date","close") if normalized_view else compare
        y = "change_pct" if normalized_view else "close"
        fig = go.Figure()
        for cid, group in plot.groupby("coin_id"):
            symbol = next((a["symbol"] for a in assets if a["coin_id"] == cid), cid)
            fig.add_trace(go.Scatter(x=group.date,y=group[y],mode="lines",name=symbol))
        fig.update_layout(template="plotly_dark",paper_bgcolor="#111b2c",plot_bgcolor="#111b2c",height=360,yaxis_title="% desde inicio" if normalized_view else "USD")
        st.plotly_chart(fig,width="stretch")

    ranking = []
    for a in assets:
        s = data["snapshots"].get(a["coin_id"], {})
        ranking.append({"Moneda":a.get("name"),"Símbolo":a.get("symbol"),"Ranking":a.get("rank"),"Precio USD":s.get("price"),"24 h %":s.get("price_change_1d"),"7 d %":s.get("price_change_1w"),"Capitalización USD":s.get("market_cap"),"Volumen 24 h USD":s.get("volume_24h")})
    ranking_df = pd.DataFrame(ranking)
    st.subheader("Ranking de mercado")
    st.dataframe(ranking_df,width="stretch",hide_index=True)
    caps = ranking_df.dropna(subset=["Capitalización USD"])
    caps = caps[caps["Capitalización USD"] > 0]
    if len(caps):
        pie = go.Figure(go.Pie(labels=caps["Símbolo"],values=caps["Capitalización USD"],hole=.56))
        pie.update_layout(title="Distribución de capitalización de la selección",template="plotly_dark",paper_bgcolor="#111b2c",height=360)
        st.plotly_chart(pie,width="stretch")
        st.caption("Participación entre los activos visibles, no de todo el mercado mundial. Una observación por activo.")

with tab_predictive:
    st.header("Pronóstico de precios y tendencias")
    horizon_label = st.select_slider("Horizonte",options=list(HORIZONS),value="3 meses")
    horizon = HORIZONS[horizon_label]
    mode = st.radio("Vista",["Moneda individual","Comparación de monedas"],horizontal=True)
    forecasts = data["forecasts"]
    if mode == "Moneda individual":
        run = data["runs"].get(coin_id)
        forecast = forecasts[forecasts.coin_id == coin_id].head(horizon)
        result = forecast_stats(forecast,float((run or {}).get("base_price") or 0),horizon)
        if not run or not result:
            st.info("Este activo aún no tiene pronóstico. Ejecuta la carga y el entrenamiento.")
        else:
            model_metric = data["metrics"].get(coin_id, {})
            if model_metric.get("beats_zero_return_baseline") is False:
                st.warning("Modelo experimental: en validación temporal XGBoost tuvo más error que pronosticar retorno cero. No uses estas fechas como señal de compra o venta.")
            if result["days_available"] < horizon:
                st.warning(f"Solo hay {result['days_available']} días disponibles para un horizonte solicitado de {horizon}.")
            base_date = pd.to_datetime(run["base_date"], utc=True)
            generated = pd.to_datetime(run["generated_at"], utc=True)
            st.caption(f"Base: {base_date:%Y-%m-%d} · Generado: {generated:%Y-%m-%d %H:%M UTC} · XGBoost recursivo")
            if (pd.Timestamp.now(tz="UTC") - generated).days > 10:
                st.warning("Pronóstico con más de 10 días desde su generación; considera reentrenar.")
            a,b,c,d = st.columns(4)
            a.metric("Precio proyectado",money(result["end_price"]))
            b.metric("Variación estimada",f"{result['variation_pct']:+.2f} %")
            c.metric("Tendencia",result["trend"])
            d.metric("Banda P10–P90 aprox.",f"{money(result['p10'])} – {money(result['p90'])}")
            a,b = st.columns(2)
            a.metric("Menor cierre proyectado · posible compra",f"{result['buy']['forecast_date']:%Y-%m-%d} · {money(result['buy']['predicted_price'])}")
            b.metric("Mayor cierre proyectado · posible venta",f"{result['sell']['forecast_date']:%Y-%m-%d} · {money(result['sell']['predicted_price'])}")
            st.plotly_chart(line_chart(forecast,"forecast_date","predicted_price",f"Trayectoria proyectada · {asset['symbol']}",shade=True),width="stretch")
            st.caption("Las fechas extremas y bandas son estimaciones del modelo, no señales ciertas de operación. La incertidumbre aumenta con el horizonte.")
    else:
        defaults = [x for x in labels if x.split(" · ")[0] in ("BTC","ETH","SOL","BNB")]
        chosen = st.multiselect("Monedas para comparar",list(labels),default=defaults,key="forecast_compare")
        selected = [labels[x] for x in chosen]
        subset = forecasts[forecasts.coin_id.isin(selected)].sort_values("forecast_date").groupby("coin_id",group_keys=False).head(horizon).copy()
        if subset.empty:
            st.info("No hay pronósticos para la selección.")
        else:
            relative = st.toggle("Comparar variación porcentual",value=True,key="forecast_relative")
            plot = normalized(subset,"forecast_date","predicted_price") if relative else subset
            fig = go.Figure()
            for cid,group in plot.groupby("coin_id"):
                symbol = next((a["symbol"] for a in assets if a["coin_id"] == cid),cid)
                fig.add_trace(go.Scatter(x=group.forecast_date,y=group["change_pct" if relative else "predicted_price"],mode="lines",name=symbol))
            fig.update_layout(template="plotly_dark",paper_bgcolor="#111b2c",plot_bgcolor="#111b2c",height=420,yaxis_title="% desde primer día" if relative else "USD")
            st.plotly_chart(fig,width="stretch")
            summary = []
            for cid, group in subset.groupby("coin_id"):
                run = data["runs"].get(cid, {})
                info = forecast_stats(group,float(run.get("base_price") or 0),horizon)
                if info:
                    summary.append({"Moneda":next((a["symbol"] for a in assets if a["coin_id"] == cid),cid),"Fecha final":info["end"].date(),"Precio proyectado USD":info["end_price"],"Variación %":round(info["variation_pct"],2),"Tendencia":info["trend"],"Días disponibles":info["days_available"]})
            st.dataframe(pd.DataFrame(summary).sort_values("Variación %",ascending=False),width="stretch",hide_index=True)

with tab_ai:
    st.header("Escenarios para la toma de decisiones")
    st.caption("Gemini interpreta evidencia cuantitativa del pronóstico vigente. El texto se genera al pulsar el botón y queda trazable en MongoDB.")
    run = data["runs"].get(coin_id)
    forecast = data["forecasts"][data["forecasts"].coin_id == coin_id].head(horizon)
    result = forecast_stats(forecast,float((run or {}).get("base_price") or 0),horizon)
    if data["demo"]:
        st.info("La IA requiere datos reales y una clave GEMINI_API_KEY. Está desactivada en demostración.")
    elif not run or not result:
        st.info("Primero genera un pronóstico real para esta moneda.")
    else:
        q = data["quality"].get(coin_id,{})
        m = data["metrics"].get(coin_id,{})
        if m.get("beats_zero_return_baseline") is False:
            st.warning("El modelo no superó el baseline en validación. Gemini analizará escenarios con esta limitación y debe priorizar observación o espera.")
        evidence = {"coin_id":coin_id,"symbol":asset.get("symbol"),"horizon_days":horizon,"forecast_run_id":run["forecast_run_id"],"base_date":str(run.get("base_date")),"base_price_usd":round(float(run["base_price"]),8),"end_date":str(result["end"]),"central_price_usd":round(result["end_price"],8),"variation_pct":round(result["variation_pct"],3),"p10_usd":round(result["p10"],8),"p90_usd":round(result["p90"],8),"coverage_pct":q.get("requested_window_coverage_pct"),"model_mae":m.get("mae_log_return"),"baseline_mae":m.get("baseline_mae_log_return"),"beats_baseline":m.get("beats_zero_return_baseline"),"model_trained_at":str(m.get("trained_at"))}
        a,b,c = st.columns(3)
        a.metric("Escenario bajista P10",money(result["p10"]))
        b.metric("Escenario central",money(result["end_price"]))
        c.metric("Escenario alcista P90",money(result["p90"]))
        try:
            recommendation = cached_recommendation(evidence)
        except Exception as exc:
            recommendation = None
            st.warning(f"No se pudo consultar el análisis guardado: {exc}")
        if st.button("Generar análisis con Gemini",type="primary",disabled=not os.getenv("GEMINI_API_KEY")):
            try:
                with st.spinner("Gemini analiza el pronóstico y sus riesgos…"):
                    recommendation = generate_recommendation(evidence)
            except Exception as exc:
                st.error(f"Gemini no pudo generar el análisis: {exc}")
        if not os.getenv("GEMINI_API_KEY"):
            st.info("Configura GEMINI_API_KEY en .env o en los secretos de Streamlit para activar la IA.")
        if recommendation:
            answer = recommendation["response"]
            a,b = st.columns(2)
            a.metric("Postura para evaluar",answer["postura"].replace("_"," "))
            b.metric("Riesgo del pronóstico",answer["riesgo"])
            st.write(answer["resumen"])
            for title,key in [("Si el mercado sube","escenario_alcista"),("Escenario central","escenario_base"),("Si el mercado cae","escenario_bajista")]:
                with st.expander(title,expanded=key=="escenario_base"):
                    st.write(answer[key])
            st.markdown("**Señales a vigilar**")
            for signal in answer["senales_a_vigilar"]:
                st.write("• "+signal)
            st.warning(answer["limitaciones"])
            st.caption(f"Modelo IA: {recommendation['model']} · Generado: {recommendation['generated_at']} · Run: {run['forecast_run_id']}")
        with st.expander("Evidencia enviada a Gemini"):
            st.json(evidence)

with tab_quality:
    st.header("Estado de la extracción y los modelos")
    rows = []
    for a in assets:
        cid = a["coin_id"]
        q = data["quality"].get(cid,{})
        m = data["metrics"].get(cid,{})
        run = data["runs"].get(cid,{})
        rows.append({"Moneda":a.get("symbol"),"Filas históricas":q.get("actual_rows"),"Cobertura solicitada %":q.get("requested_window_coverage_pct"),"Estado":q.get("status"),"MAE modelo":m.get("mae_log_return"),"MAE baseline":m.get("baseline_mae_log_return"),"Supera baseline":m.get("beats_zero_return_baseline"),"Pronóstico días":run.get("forecast_days"),"Última fecha":q.get("last_available_date")})
    st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)
    st.caption("Cobertura solicitada = días observados / ventana configurada de extracción. Un modelo que no supera el retorno cero debe interpretarse con especial cautela.")
    if not data["demo"]:
        st.info("Para auditar duplicados, fechas futuras, precios inválidos y huecos de toda la base: python scripts/audit_data.py")
