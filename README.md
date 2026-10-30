# CryptoAnalytics · Grupo 07

Aplicación Streamlit con tres vistas: descriptiva, predictiva (XGBoost) y prescriptiva (escenarios redactados por Gemini). La fuente de verdad es MongoDB Atlas. CoinStats aporta snapshots; Binance Spot aporta cierres OHLCV diarios; Mobula es opcional y solo se incorpora cuando su API autoriza las consultas.

## Inicio local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Completa `.env` con tus claves y URI de Atlas. No publiques `.env`. El nombre de base configurado en este proyecto es `crypto_analytics_powerbi`; puedes cambiarlo con `MONGODB_DB_NAME`. Para la primera carga:

```powershell
python scripts/bootstrap.py
python scripts/audit_data.py
streamlit run app.py
```

La carga inicial puede tardar varios minutos porque consulta hasta cinco años de historia y valida cada modelo en varios cortes temporales. Si la base está vacía o inaccesible, Streamlit muestra datos ficticios con una advertencia visible. La vista de IA queda deshabilitada en ese modo.

## Operación

```powershell
python scripts/refresh.py            # mercado y noticias
python scripts/refresh.py --daily    # histórico incremental y capa diaria
python scripts/refresh.py --train    # lo anterior, XGBoost y pronósticos
python scripts/audit_data.py         # cobertura, huecos e integridad
```

`scripts/bootstrap.py` y cada actualización llaman `ensure_indexes()` antes de escribir. Hay índices únicos para activos, series diarias, features y puntos de pronóstico, más índices de búsqueda por moneda y fecha. `market_daily` guarda una fila por activo/día. La prioridad de fuente es Mobula OHLCV, Binance OHLCV y, al final, historial de precio Mobula. Los días sin observación no se rellenan para el modelo.

Mobula puede responder HTTP 401/403 según la clave o el plan. En ese caso la carga sigue con Binance y el aviso queda en `pipeline_runs`; algunos activos sin par `USDT` no tendrán historia ni modelo. CoinStats debe devolver un catálogo válido antes de modificar el Top N.

## Interpretación

- **Descriptivo:** snapshot más reciente, serie de cierres, ranking, capitalización y comparación porcentual normalizada. Las métricas de periodo usan el primer y último cierre observados y advierten cobertura parcial.
- **Predictivo:** XGBoost pronostica el retorno logarítmico del día siguiente; la trayectoria se genera recursivamente hasta `FORECAST_DAYS`. La banda P10–P90 es una aproximación con la desviación de errores de validación temporal. Las fechas de menor/mayor cierre proyectado son extremos del escenario central, no oportunidades garantizadas.
- **Prescriptivo:** Gemini recibe solo cifras de un `forecast_run_id`, métricas de validación y cobertura. Devuelve una postura para evaluar, riesgo, tres escenarios condicionales y señales a vigilar. El resultado estructurado se guarda por huella de evidencia en `ai_recommendations`; una nueva corrida o horizonte produce nueva evidencia. Requiere `GEMINI_API_KEY`. No se llama a Gemini en cada recarga de página.
- **Calidad:** la pestaña muestra cobertura de la ventana solicitada y si XGBoost mejora el baseline de retorno cero. Una cobertura alta no garantiza una predicción precisa.

El horizonte de un año requiere `FORECAST_DAYS=365` y volver a generar pronósticos. Si solo hay 360 días, la interfaz lo advierte.

## Streamlit Community Cloud

Publica el repositorio en GitHub y crea una aplicación cuyo archivo principal sea `app.py`. En los secretos de la aplicación configura `MONGODB_URI`, `MONGODB_DB_NAME`, `GEMINI_API_KEY` y opcionalmente `GEMINI_MODEL`. La interfaz lee estas claves con `st.secrets`; el workflow de extracción usa secretos de **GitHub Actions** por separado: `MONGODB_URI`, `MONGODB_DB_NAME`, `COINSTATS_API_KEY`, `MOBULA_API_KEY`. Atlas debe permitir conexiones desde los entornos de ejecución elegidos. No se incluyen credenciales en el repositorio.

El workflow `.github/workflows/refresh.yml` actualiza mercado cuatro veces al día, historia una vez y modelos semanalmente. También admite ejecución manual. Configura los secretos antes de habilitar las ejecuciones programadas.

## Estructura

`app.py` contiene la UI; `src/dashboard/` carga datos, calcula métricas y genera análisis de IA; `src/etl/` extrae y cura; `src/ml/` entrena; `scripts/` ejecuta la carga, actualización y auditoría.

Documentación: [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [secretos de Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), [salida estructurada de Gemini](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).
