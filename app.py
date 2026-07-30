"""
Observatorio Antioquia — Dashboard de Data Science
====================================================
Consolida en un solo tablero: datos geoespaciales sobre mapas reales,
estadísticas, tablas, series de tiempo y un analista de IA (Groq + Llama
3.3 70B) que razona sobre los datos filtrados en pantalla.

Todos los datos son simulados (ver data_generator.py) pero construidos con
relaciones causales reales entre variables, para que los análisis de
correlación, series de tiempo y el propio LLM tengan algo genuino que decir.
"""

import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import generar_todo, TIPOS_EVENTO

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Observatorio Antioquia · Data Science Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Tema visual — "carta topográfica nocturna"
#   fondo:   #0B1F2A (tinta profunda)  |  superficie: #123244
#   acento cálido (oro topográfico):   #E3B23C
#   acento frío (agua):                #4FA8A0
#   texto:   #EDEBE3 (papel viejo)
# Tipografía: Spectral (display/serif cartográfico) + IBM Plex Mono (datos)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500&display=swap');

:root {
    --bg: #0B1F2A;
    --surface: #123244;
    --surface-2: #163C50;
    --gold: #E3B23C;
    --teal: #4FA8A0;
    --paper: #EDEBE3;
    --muted: #9FB3BE;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 0%, #123244 0%, #0B1F2A 55%) fixed;
    color: var(--paper);
}

h1, h2, h3 { font-family: 'Spectral', serif !important; color: var(--paper) !important; letter-spacing: 0.2px; }

h1 { border-bottom: 1px solid rgba(227,178,60,0.35); padding-bottom: 0.4rem; }

/* Eyebrow / kicker sobre el título principal */
.kicker {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--gold);
    letter-spacing: 3px;
    font-size: 0.75rem;
    text-transform: uppercase;
    margin-bottom: -0.6rem;
}

/* Tarjetas tipo "leyenda de mapa" */
div[data-testid="stMetric"] {
    background: linear-gradient(160deg, var(--surface) 0%, var(--surface-2) 100%);
    border: 1px solid rgba(227,178,60,0.25);
    border-left: 3px solid var(--gold);
    border-radius: 6px;
    padding: 0.9rem 1rem 0.7rem 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 1px;}
div[data-testid="stMetricValue"] { color: var(--paper) !important; font-family: 'Spectral', serif !important; }

/* Contorno topográfico decorativo bajo el header */
.contour {
    height: 26px;
    background-image: repeating-radial-gradient(circle at 50% 250%, transparent 0, transparent 18px, rgba(79,168,160,0.18) 19px, transparent 20px);
    background-size: 140px 140px;
    margin-bottom: 0.6rem;
    opacity: 0.8;
}

section[data-testid="stSidebar"] {
    background: #0E2531;
    border-right: 1px solid rgba(227,178,60,0.15);
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border-radius: 6px 6px 0 0;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom: 2px solid var(--gold); }

.streamlit-expanderHeader { font-family: 'IBM Plex Mono', monospace; }

/* Chat IA */
.stChatMessage { background: var(--surface); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);}

hr { border-color: rgba(227,178,60,0.2); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Carga de datos (simulados, cacheados)
# ---------------------------------------------------------------------------
SEED = 42
municipios_df, series_df, eventos_df = generar_todo(SEED)
series_df["fecha"] = pd.to_datetime(series_df["fecha"])
eventos_df["fecha"] = pd.to_datetime(eventos_df["fecha"])

# ---------------------------------------------------------------------------
# Sidebar — filtros globales + configuración de Groq
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='kicker'>OBSERVATORIO · ANTIOQUIA</div>", unsafe_allow_html=True)
    st.title("Panel de control")

    subregiones = sorted(municipios_df["subregion"].unique().tolist())
    sel_subregiones = st.multiselect("Subregión", subregiones, default=subregiones)

    municipios_disponibles = municipios_df[municipios_df["subregion"].isin(sel_subregiones)]["municipio"].tolist()
    sel_municipios = st.multiselect(
        "Municipios", municipios_disponibles, default=municipios_disponibles
    )

    fecha_min, fecha_max = series_df["fecha"].min(), series_df["fecha"].max()
    rango_fechas = st.slider(
        "Rango de fechas",
        min_value=fecha_min.to_pydatetime(),
        max_value=fecha_max.to_pydatetime(),
        value=(fecha_min.to_pydatetime(), fecha_max.to_pydatetime()),
        format="MMM YYYY",
    )

    categorias_disp = sorted(eventos_df["categoria"].unique().tolist())
    sel_categorias = st.multiselect("Categoría de evento", categorias_disp, default=categorias_disp)

    st.divider()
    st.markdown("<div class='kicker'>ANALISTA IA · GROQ</div>", unsafe_allow_html=True)
    groq_api_key = st.text_input(
        "GROQ API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Se usa solo en esta sesión, no se almacena. Consíguela en console.groq.com",
    )
    modelo_groq = st.selectbox(
        "Modelo",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
    )
    st.caption("El modelo responde usando como contexto los datos ya filtrados en este panel.")

# ---------------------------------------------------------------------------
# Aplicar filtros a todas las tablas
# ---------------------------------------------------------------------------
if not sel_municipios:
    st.warning("Selecciona al menos un municipio en el panel lateral para ver el dashboard.")
    st.stop()

f_ini, f_fin = pd.Timestamp(rango_fechas[0]), pd.Timestamp(rango_fechas[1])

mun_f = municipios_df[municipios_df["municipio"].isin(sel_municipios)].copy()
series_f = series_df[
    (series_df["municipio"].isin(sel_municipios)) &
    (series_df["fecha"].between(f_ini, f_fin))
].copy()
eventos_f = eventos_df[
    (eventos_df["municipio"].isin(sel_municipios)) &
    (eventos_df["fecha"].between(f_ini, f_fin)) &
    (eventos_df["categoria"].isin(sel_categorias))
].copy()

# Tabla resumen por municipio (para mapa, tablas y correlaciones)
resumen_mun = (
    series_f.groupby("municipio")
    .agg(
        inversion_total_mcop=("inversion_publica_mcop", "sum"),
        eventos_total=("eventos_seguridad", "sum"),
        empleo_prom=("empleo_formal_tasa", "mean"),
        casos_salud_total=("casos_salud", "sum"),
    )
    .reset_index()
    .merge(mun_f, on="municipio", how="left")
)
resumen_mun["eventos_por_10k_hab"] = (resumen_mun["eventos_total"] / resumen_mun["poblacion_base"] * 10000).round(2)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='kicker'>DATA SCIENCE · GEOANALÍTICA · SERIES DE TIEMPO · IA</div>", unsafe_allow_html=True)
st.title("🗺️ Observatorio Antioquia")
st.markdown("<div class='contour'></div>", unsafe_allow_html=True)
st.caption(
    f"Datos simulados con relaciones causales (IPM → eventos → inversión → empleo → salud) · "
    f"{len(sel_municipios)} municipios · {f_ini:%b %Y} – {f_fin:%b %Y}"
)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Población cubierta", f"{int(resumen_mun['poblacion_base'].sum()):,}".replace(",", "."))
k2.metric("IPM promedio", f"{resumen_mun['ipm'].mean():.1f}")
k3.metric("Inversión pública total", f"${resumen_mun['inversion_total_mcop'].sum():,.0f} MCOP".replace(",", "."))
k4.metric("Eventos registrados", f"{int(resumen_mun['eventos_total'].sum()):,}".replace(",", "."))
k5.metric("Empleo formal (prom.)", f"{resumen_mun['empleo_prom'].mean():.1f}%")

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_mapa, tab_series, tab_tablas, tab_correl, tab_ia = st.tabs(
    ["🗺️ Mapa & Resumen", "📈 Series de tiempo", "📊 Tablas & datos", "🔗 Correlaciones", "🤖 Analista IA (Groq)"]
)

# =========================== TAB 1 — MAPA ==================================
with tab_mapa:
    col_map, col_side = st.columns([2.1, 1])

    with col_map:
        st.subheader("Mapa geoespacial real — eventos por municipio")
        metrica_mapa = st.radio(
            "Tamaño de burbuja según:",
            ["eventos_total", "inversion_total_mcop", "casos_salud_total", "ipm"],
            format_func=lambda x: {
                "eventos_total": "Eventos totales",
                "inversion_total_mcop": "Inversión pública",
                "casos_salud_total": "Casos de salud",
                "ipm": "IPM (pobreza)",
            }[x],
            horizontal=True,
        )
        fig_map = px.scatter_mapbox(
            resumen_mun,
            lat="lat", lon="lon",
            size=metrica_mapa,
            color="ipm",
            color_continuous_scale=["#4FA8A0", "#E3B23C", "#C1440E"],
            size_max=42,
            zoom=6.4,
            hover_name="municipio",
            hover_data={
                "subregion": True, "ipm": ":.1f", "eventos_total": True,
                "inversion_total_mcop": ":.0f", "empleo_prom": ":.1f",
                "lat": False, "lon": False,
            },
            mapbox_style="carto-darkmatter",
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#EDEBE3",
            coloraxis_colorbar=dict(title="IPM"),
            height=520,
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("Mapa real (OpenStreetMap / Carto) — tamaño = métrica elegida, color = Índice de Pobreza Multidimensional (IPM).")

    with col_side:
        st.subheader("Eventos individuales geolocalizados")
        st.dataframe(
            eventos_f.sort_values("fecha", ascending=False)[
                ["fecha", "municipio", "tipo_evento", "gravedad"]
            ].head(300),
            use_container_width=True,
            height=400,
            hide_index=True,
        )
        st.caption(f"{len(eventos_f):,} eventos individuales en el filtro actual.".replace(",", "."))

    st.divider()
    st.subheader("Ranking de municipios")
    rank_metric = st.selectbox(
        "Ordenar por",
        ["ipm", "eventos_por_10k_hab", "inversion_total_mcop", "empleo_prom", "casos_salud_total"],
        format_func=lambda x: {
            "ipm": "IPM", "eventos_por_10k_hab": "Eventos por 10k hab.",
            "inversion_total_mcop": "Inversión total", "empleo_prom": "Empleo formal promedio",
            "casos_salud_total": "Casos de salud",
        }[x],
    )
    fig_rank = px.bar(
        resumen_mun.sort_values(rank_metric, ascending=True),
        x=rank_metric, y="municipio", orientation="h",
        color="subregion",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_rank.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EDEBE3", height=460, legend_title="Subregión",
    )
    st.plotly_chart(fig_rank, use_container_width=True)

# ======================= TAB 2 — SERIES DE TIEMPO ==========================
with tab_series:
    st.subheader("Evolución mensual")
    metrica_ts = st.selectbox(
        "Variable",
        ["eventos_seguridad", "inversion_publica_mcop", "empleo_formal_tasa", "casos_salud"],
        format_func=lambda x: {
            "eventos_seguridad": "Eventos de seguridad",
            "inversion_publica_mcop": "Inversión pública (MCOP)",
            "empleo_formal_tasa": "Empleo formal (%)",
            "casos_salud": "Casos de atención en salud",
        }[x],
        key="ts_metric",
    )

    modo = st.radio("Vista", ["Comparar municipios", "Agregado (total/promedio)"], horizontal=True)

    if modo == "Comparar municipios":
        fig_ts = px.line(
            series_f.sort_values("fecha"),
            x="fecha", y=metrica_ts, color="municipio",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
    else:
        agg_fn = "mean" if metrica_ts == "empleo_formal_tasa" else "sum"
        agregada = series_f.groupby("fecha", as_index=False)[metrica_ts].agg(agg_fn)
        fig_ts = px.area(agregada, x="fecha", y=metrica_ts)
        fig_ts.update_traces(line_color="#E3B23C", fillcolor="rgba(227,178,60,0.25)")

    fig_ts.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EDEBE3", height=460, xaxis_title="", legend_title="Municipio",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.divider()
    st.markdown("##### Comparación multi-variable normalizada (índice 100 = inicio del periodo)")
    base = series_f.groupby("fecha", as_index=False)[
        ["eventos_seguridad", "inversion_publica_mcop", "empleo_formal_tasa", "casos_salud"]
    ].sum()
    base = base.sort_values("fecha")
    norm = base.copy()
    for col in ["eventos_seguridad", "inversion_publica_mcop", "empleo_formal_tasa", "casos_salud"]:
        primer_valor = norm[col].iloc[0] if norm[col].iloc[0] != 0 else 1
        norm[col] = norm[col] / primer_valor * 100

    fig_norm = go.Figure()
    nombres = {
        "eventos_seguridad": "Eventos de seguridad", "inversion_publica_mcop": "Inversión pública",
        "empleo_formal_tasa": "Empleo formal", "casos_salud": "Casos de salud",
    }
    colores = {"eventos_seguridad": "#C1440E", "inversion_publica_mcop": "#E3B23C",
               "empleo_formal_tasa": "#4FA8A0", "casos_salud": "#8FB8DE"}
    for col, nombre in nombres.items():
        fig_norm.add_trace(go.Scatter(x=norm["fecha"], y=norm[col], name=nombre, line=dict(color=colores[col])))
    fig_norm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EDEBE3", height=420, yaxis_title="Índice (base 100)",
    )
    st.plotly_chart(fig_norm, use_container_width=True)
    st.caption("Permite ver, en la misma escala, cómo se mueven juntas (o en contra) las 4 variables clave del sistema.")

# ========================= TAB 3 — TABLAS ==================================
with tab_tablas:
    st.subheader("Datos estructurados")

    sub_t1, sub_t2, sub_t3 = st.tabs(["Municipios (maestro)", "Series de tiempo (mensual)", "Eventos (detalle)"])

    with sub_t1:
        st.dataframe(
            resumen_mun[[
                "municipio", "subregion", "tipo", "poblacion_base", "ipm",
                "indice_institucionalidad", "inversion_total_mcop", "eventos_total",
                "eventos_por_10k_hab", "empleo_prom", "casos_salud_total",
            ]].sort_values("municipio"),
            use_container_width=True, hide_index=True,
            column_config={
                "ipm": st.column_config.ProgressColumn("IPM", min_value=0, max_value=65, format="%.1f"),
                "indice_institucionalidad": st.column_config.ProgressColumn("Institucionalidad", min_value=0, max_value=100, format="%.1f"),
                "poblacion_base": st.column_config.NumberColumn("Población", format="%d"),
                "inversion_total_mcop": st.column_config.NumberColumn("Inversión (MCOP)", format="%.0f"),
                "empleo_prom": st.column_config.NumberColumn("Empleo % (prom)", format="%.1f"),
            },
        )
        st.download_button(
            "⬇️ Descargar CSV — Municipios",
            resumen_mun.to_csv(index=False).encode("utf-8"),
            "municipios_resumen.csv", "text/csv",
        )

    with sub_t2:
        st.dataframe(
            series_f.sort_values(["municipio", "fecha"]),
            use_container_width=True, hide_index=True, height=420,
        )
        st.download_button(
            "⬇️ Descargar CSV — Series de tiempo",
            series_f.to_csv(index=False).encode("utf-8"),
            "series_tiempo.csv", "text/csv",
        )

    with sub_t3:
        st.dataframe(
            eventos_f.sort_values("fecha", ascending=False),
            use_container_width=True, hide_index=True, height=420,
        )
        st.download_button(
            "⬇️ Descargar CSV — Eventos detalle",
            eventos_f.to_csv(index=False).encode("utf-8"),
            "eventos_detalle.csv", "text/csv",
        )

# ======================= TAB 4 — CORRELACIONES ==============================
with tab_correl:
    st.subheader("Matriz de correlación entre variables")
    cols_num = ["ipm", "indice_institucionalidad", "poblacion_base", "inversion_total_mcop",
                "eventos_total", "eventos_por_10k_hab", "empleo_prom", "casos_salud_total"]
    corr = resumen_mun[cols_num].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale=["#4FA8A0", "#123244", "#C1440E"],
        aspect="auto", zmin=-1, zmax=1,
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#EDEBE3", height=480,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### IPM vs. Eventos de seguridad")
        fig_s1 = px.scatter(
            resumen_mun, x="ipm", y="eventos_por_10k_hab", size="poblacion_base",
            color="subregion", trendline="ols", hover_name="municipio",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_s1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#EDEBE3", height=420)
        st.plotly_chart(fig_s1, use_container_width=True)
    with c2:
        st.markdown("##### Inversión pública vs. Empleo formal")
        fig_s2 = px.scatter(
            resumen_mun, x="inversion_total_mcop", y="empleo_prom", size="poblacion_base",
            color="subregion", trendline="ols", hover_name="municipio",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_s2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#EDEBE3", height=420)
        st.plotly_chart(fig_s2, use_container_width=True)

    st.caption(
        "Estas relaciones no son azar: el motor de simulación las construyó explícitamente "
        "(ver `data_generator.py`). Nota de método: la inversión pública se asigna de forma "
        "contracíclica (más recursos donde el IPM es más alto), por eso su correlación *entre "
        "municipios* con el empleo puede verse negativa — es un efecto de selección/focalización, "
        "no causalidad inversa. El efecto positivo real de la inversión sobre el empleo está "
        "modelado con rezago *dentro* de cada municipio (ver pestaña Series de tiempo). Es un buen "
        "ejemplo de por qué correlación cruzada ≠ causalidad."
    )

# ========================= TAB 5 — ANALISTA IA ==============================
with tab_ia:
    st.subheader("Analista IA · Groq + Llama 3.3 70B")
    st.caption(
        "El modelo recibe como contexto un resumen numérico de los datos ya filtrados en el panel "
        "lateral (no el dataset completo), y responde en español con hallazgos, hipótesis y sugerencias."
    )

    def construir_contexto() -> str:
        top_ipm = resumen_mun.sort_values("ipm", ascending=False).head(3)["municipio"].tolist()
        top_eventos = resumen_mun.sort_values("eventos_por_10k_hab", ascending=False).head(3)["municipio"].tolist()
        top_inversion = resumen_mun.sort_values("inversion_total_mcop", ascending=False).head(3)["municipio"].tolist()
        corr_ipm_ev = resumen_mun[["ipm", "eventos_por_10k_hab"]].corr().iloc[0, 1]
        corr_inv_emp = resumen_mun[["inversion_total_mcop", "empleo_prom"]].corr().iloc[0, 1]

        contexto = textwrap.dedent(f"""
        CONTEXTO DE DATOS (filtro activo del usuario en el dashboard):
        - Periodo: {f_ini:%Y-%m} a {f_fin:%Y-%m}
        - Municipios incluidos ({len(sel_municipios)}): {", ".join(sel_municipios)}
        - Población total cubierta: {int(resumen_mun['poblacion_base'].sum()):,}
        - IPM promedio del grupo: {resumen_mun['ipm'].mean():.1f} (0=sin pobreza, 100=máxima)
        - Municipios con mayor IPM: {", ".join(top_ipm)}
        - Municipios con más eventos por 10k habitantes: {", ".join(top_eventos)}
        - Municipios con mayor inversión pública acumulada: {", ".join(top_inversion)}
        - Inversión pública total del periodo: {resumen_mun['inversion_total_mcop'].sum():,.0f} millones COP
        - Eventos de seguridad totales: {int(resumen_mun['eventos_total'].sum())}
        - Empleo formal promedio: {resumen_mun['empleo_prom'].mean():.1f}%
        - Casos de atención en salud totales: {int(resumen_mun['casos_salud_total'].sum())}
        - Correlación IPM vs eventos por 10k hab: {corr_ipm_ev:.2f}
        - Correlación inversión pública vs empleo formal: {corr_inv_emp:.2f}

        Tabla resumen por municipio (CSV):
        {resumen_mun[["municipio","subregion","ipm","eventos_total","eventos_por_10k_hab","inversion_total_mcop","empleo_prom","casos_salud_total"]].round(1).to_csv(index=False)}
        """).strip()
        return contexto

    SYSTEM_PROMPT = (
        "Eres un analista senior de datos y política pública para el departamento de Antioquia, Colombia. "
        "Respondes SIEMPRE en español, de forma clara, estructurada (usa viñetas cuando ayude) y basada "
        "estrictamente en los datos de contexto que se te entregan en cada turno. Todos los datos son "
        "simulados con fines de demostración de un dashboard de ciencia de datos: acláralo si el usuario "
        "pregunta si son datos reales. Cuando detectes correlaciones, sé cuidadoso en no afirmar causalidad "
        "sin evidencia adicional, pero puedes proponer hipótesis razonables. Sé específico citando cifras y "
        "nombres de municipios del contexto."
    )

    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []

    for msg in st.session_state.chat_historial:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    colb1, colb2 = st.columns([1, 5])
    with colb1:
        if st.button("🧹 Limpiar chat"):
            st.session_state.chat_historial = []
            st.rerun()
    with colb2:
        sugerencias = st.selectbox(
            "Preguntas sugeridas",
            [
                "— Elige una pregunta rápida —",
                "¿Qué municipios requieren atención prioritaria y por qué?",
                "Explica la relación entre inversión pública y empleo formal en este grupo.",
                "¿Qué hipótesis explicarían los municipios con más eventos por habitante?",
                "Dame 3 recomendaciones de política pública basadas en estos datos.",
            ],
            label_visibility="collapsed",
        )

    prompt_usuario = st.chat_input("Pregúntale al analista IA sobre los datos filtrados...")
    if sugerencias != "— Elige una pregunta rápida —" and not prompt_usuario:
        prompt_usuario = sugerencias

    if prompt_usuario:
        if not groq_api_key:
            st.error("Ingresa tu GROQ API Key en el panel lateral para usar el analista IA.")
        else:
            st.session_state.chat_historial.append({"role": "user", "content": prompt_usuario})
            with st.chat_message("user"):
                st.markdown(prompt_usuario)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("▌ Analizando datos filtrados...")
                try:
                    from groq import Groq

                    client = Groq(api_key=groq_api_key)
                    mensajes = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "system", "content": construir_contexto()},
                    ] + st.session_state.chat_historial[-8:]

                    respuesta_stream = client.chat.completions.create(
                        model=modelo_groq,
                        messages=mensajes,
                        temperature=0.4,
                        max_tokens=1024,
                        stream=True,
                    )
                    texto_completo = ""
                    for chunk in respuesta_stream:
                        delta = chunk.choices[0].delta.content or ""
                        texto_completo += delta
                        placeholder.markdown(texto_completo + "▌")
                    placeholder.markdown(texto_completo)
                    st.session_state.chat_historial.append({"role": "assistant", "content": texto_completo})
                except Exception as e:
                    placeholder.error(f"Error al llamar a la API de Groq: {e}")

    with st.expander("📎 Ver el contexto exacto que recibe el modelo"):
        st.code(construir_contexto(), language="text")

st.divider()
st.caption(
    "Observatorio Antioquia · Dashboard de demostración con datos 100% simulados · "
    f"Generado el {datetime.now():%Y-%m-%d %H:%M}"
)
