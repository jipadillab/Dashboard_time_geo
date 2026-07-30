"""
data_generator.py
------------------
Motor de datos simulados para el dashboard "Observatorio Antioquia".

Todos los datos son SINTÉTICOS (no reales), pero se generan con relaciones
causales explícitas entre variables para que el dashboard tenga sentido
analítico real:

    Pobreza (IPM) ──► Eventos de seguridad ──► (afecta) Empleo formal
         │                                            ▲
         └────────────► Inversión pública ────────────┘
                              │
                              ▼
                     Casos de atención en salud

La idea: un municipio con IPM alto tiende a tener más eventos, lo cual
reduce la inversión privada y el empleo formal, lo cual a su vez presiona
la demanda de salud pública. La inversión pública busca compensar el IPM
alto (política contracíclica) con un rezago de meses.

Todo queda cacheado con @st.cache_data para que la app no regenere datos
en cada interacción, solo cuando cambia la semilla.
"""

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Catálogo base: municipios reales de Antioquia con coordenadas reales
# ---------------------------------------------------------------------------
MUNICIPIOS = [
    # nombre, subregión, lat, lon, población base, tipo
    ("Medellín",        "Valle de Aburrá", 6.2442, -75.5812, 2560000, "Metrópoli"),
    ("Bello",           "Valle de Aburrá", 6.3373, -75.5581, 530000,  "Urbano grande"),
    ("Itagüí",          "Valle de Aburrá", 6.1719, -75.6122, 280000,  "Urbano grande"),
    ("Envigado",        "Valle de Aburrá", 6.1689, -75.5817, 240000,  "Urbano grande"),
    ("Rionegro",        "Oriente",         6.1537, -75.3746, 175000,  "Urbano medio"),
    ("Marinilla",       "Oriente",         6.1730, -75.3378, 60000,   "Urbano medio"),
    ("Santa Fe de Antioquia", "Occidente", 6.5568, -75.8272, 25000,   "Rural-turístico"),
    ("Turbo",           "Urabá",           8.0945, -76.7290, 170000,  "Urbano medio"),
    ("Apartadó",        "Urabá",           7.8828, -76.6247, 190000,  "Urbano medio"),
    ("Necoclí",         "Urabá",           8.4257, -76.7853, 65000,   "Rural"),
    ("Caucasia",        "Bajo Cauca",      7.9889, -75.1978, 120000,  "Urbano medio"),
    ("El Bagre",        "Bajo Cauca",      7.5966, -74.8072, 55000,   "Rural minero"),
    ("Segovia",         "Nordeste",        7.0794, -74.7020, 45000,   "Rural minero"),
    ("Yarumal",         "Norte",           6.9614, -75.4186, 50000,   "Rural"),
    ("Andes",           "Suroeste",        5.6584, -75.8783, 45000,   "Rural cafetero"),
    ("Puerto Berrío",   "Magdalena Medio", 6.4877, -74.4032, 45000,   "Rural fluvial"),
]

COLUMNAS_MUNICIPIO = ["municipio", "subregion", "lat", "lon", "poblacion_base", "tipo"]

FECHA_INICIO = "2023-01-01"
FECHA_FIN = "2025-12-01"  # 36 meses

TIPOS_EVENTO = [
    ("Hurto",              0.42, "Seguridad"),
    ("Riña / disturbio",   0.20, "Seguridad"),
    ("Accidente de tránsito", 0.18, "Movilidad"),
    ("Emergencia de salud pública", 0.12, "Salud"),
    ("Desplazamiento forzado", 0.05, "Humanitario"),
    ("Incidente ambiental", 0.03, "Ambiental"),
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@st.cache_data(show_spinner="Generando universo de datos simulados...")
def generar_municipios(seed: int = 42) -> pd.DataFrame:
    """Genera el índice base de pobreza multidimensional (IPM) y atributos
    estructurales de cada municipio. El IPM es la variable 'causa raíz' de
    todo el sistema."""
    rng = _rng(seed)
    df = pd.DataFrame(MUNICIPIOS, columns=COLUMNAS_MUNICIPIO)

    # IPM base: municipios rurales/mineros tienden a un IPM más alto,
    # metrópolis y urbano grande más bajo (con ruido realista)
    ipm_por_tipo = {
        "Metrópoli": 12, "Urbano grande": 18, "Urbano medio": 28,
        "Rural-turístico": 26, "Rural": 42, "Rural minero": 46,
        "Rural cafetero": 33, "Rural fluvial": 38,
    }
    df["ipm"] = df["tipo"].map(ipm_por_tipo).astype(float)
    df["ipm"] += rng.normal(0, 3.5, size=len(df))
    df["ipm"] = df["ipm"].clip(5, 65).round(1)

    # Presencia institucional (0-100): inversamente relacionada al IPM
    df["indice_institucionalidad"] = (100 - df["ipm"] * 1.1 + rng.normal(0, 4, len(df))).clip(15, 95).round(1)

    df["codigo_dane"] = [f"05{str(i).zfill(3)}" for i in range(1, len(df) + 1)]
    return df


@st.cache_data(show_spinner="Simulando series de tiempo mensuales...")
def generar_series_tiempo(seed: int = 42) -> pd.DataFrame:
    """Genera series mensuales por municipio con relaciones causales:

    eventos_seguridad  = f(IPM, estacionalidad, ruido) - f(inversion rezagada)
    inversion_publica  = f(IPM, contraciclica) + tendencia creciente + ruido
    empleo_formal_tasa  = f(inversion, - eventos) + tendencia
    casos_salud        = f(poblacion, IPM, eventos) + estacionalidad (picos en oct-dic)
    """
    rng = _rng(seed)
    municipios = generar_municipios(seed)
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="MS")

    filas = []
    for _, m in municipios.iterrows():
        ipm = m["ipm"]
        poblacion = m["poblacion_base"]

        # Inversión pública base mensual (millones COP), contracíclica al IPM
        inv_base = 80 + ipm * 6.5
        # Serie de inversión con tendencia leve creciente + estacionalidad fin de año
        t = np.arange(len(fechas))
        estacional_inv = 1 + 0.25 * (fechas.month.isin([11, 12])).astype(float)
        inversion = inv_base * (1 + 0.01 * t) * estacional_inv
        inversion += rng.normal(0, inv_base * 0.06, len(fechas))
        inversion = np.clip(inversion, 20, None)

        # Eventos de seguridad: sube con IPM, baja con inversión rezagada (lag 3 meses)
        inv_rezagada = pd.Series(inversion).shift(3).bfill().to_numpy()
        base_eventos = 4 + (ipm / 3.0) + (poblacion / 60000)
        efecto_inversion = -0.02 * (inv_rezagada - inv_base)
        estacional_ev = 1 + 0.15 * np.sin(2 * np.pi * (t % 12) / 12)
        eventos = (base_eventos + efecto_inversion) * estacional_ev
        eventos += rng.normal(0, base_eventos * 0.18, len(fechas))
        eventos = np.clip(np.round(eventos), 0, None)

        # Empleo formal (tasa %): sube con inversión, baja con eventos
        empleo = 55 - ipm * 0.35 + 0.015 * (inversion - inv_base) - 0.25 * eventos
        empleo += rng.normal(0, 1.2, len(fechas))
        empleo = np.clip(empleo, 20, 85)

        # Casos de atención en salud: sube con población, IPM y eventos; pico oct-dic
        estacional_salud = 1 + 0.2 * (fechas.month.isin([10, 11, 12, 1])).astype(float)
        casos_salud = (poblacion / 4000) * (1 + ipm / 100) * estacional_salud
        casos_salud += eventos * 1.8
        casos_salud += rng.normal(0, casos_salud.mean() * 0.08 if len(casos_salud) else 1, len(fechas))
        casos_salud = np.clip(np.round(casos_salud), 0, None)

        for i, f in enumerate(fechas):
            filas.append({
                "municipio": m["municipio"],
                "subregion": m["subregion"],
                "fecha": f,
                "inversion_publica_mcop": round(float(inversion[i]), 1),
                "eventos_seguridad": int(eventos[i]),
                "empleo_formal_tasa": round(float(empleo[i]), 1),
                "casos_salud": int(casos_salud[i]),
            })

    return pd.DataFrame(filas)


@st.cache_data(show_spinner="Geolocalizando eventos individuales...")
def generar_eventos_detalle(seed: int = 42) -> pd.DataFrame:
    """Desagrega el conteo mensual de 'eventos_seguridad' en eventos
    individuales geo-referenciados (jitter alrededor del centroide del
    municipio), con tipo y nivel de gravedad, para pintarlos en el mapa
    y alimentar la tabla de detalle."""
    rng = _rng(seed + 1)
    series = generar_series_tiempo(seed)
    municipios = generar_municipios(seed).set_index("municipio")

    tipos, pesos, categorias = zip(*TIPOS_EVENTO)
    pesos = np.array(pesos) / np.sum(pesos)

    filas = []
    eid = 0
    for _, row in series.iterrows():
        n = int(row["eventos_seguridad"])
        if n == 0:
            continue
        lat0 = municipios.loc[row["municipio"], "lat"]
        lon0 = municipios.loc[row["municipio"], "lon"]
        dias = rng.integers(1, 28, size=n)
        tipos_evento = rng.choice(len(tipos), size=n, p=pesos)
        gravedad = rng.choice(["Baja", "Media", "Alta"], size=n, p=[0.55, 0.32, 0.13])
        for i in range(n):
            eid += 1
            filas.append({
                "id_evento": eid,
                "municipio": row["municipio"],
                "subregion": row["subregion"],
                "fecha": row["fecha"] + pd.Timedelta(days=int(dias[i]) - 1),
                "tipo_evento": tipos[tipos_evento[i]],
                "categoria": categorias[tipos_evento[i]],
                "gravedad": gravedad[i],
                "lat": lat0 + rng.normal(0, 0.045),
                "lon": lon0 + rng.normal(0, 0.045),
            })
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False)
def generar_todo(seed: int = 42):
    """Punto de entrada único: retorna las 3 tablas relacionadas."""
    municipios = generar_municipios(seed)
    series = generar_series_tiempo(seed)
    eventos = generar_eventos_detalle(seed)
    return municipios, series, eventos
