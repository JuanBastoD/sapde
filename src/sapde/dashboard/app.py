"""
SAPDE — Página principal: resumen del sistema.
Ejecutar: streamlit run src/sapde/dashboard/app.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import plotly.graph_objects as go
import streamlit as st

from sapde.dashboard.utils import (
    NIVELES_COLOR,
    NIVEL_ETIQUETA,
    cargar_modelo,
    cargar_predicciones_db,
    cargar_todos_artefactos,
    obtener_engine,
)

st.set_page_config(
    page_title="SAPDE",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stSidebar"] { background-color: #1B3A6B; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p { color: #D8E4F5 !important; }
[data-testid="stSidebar"] a { color: #A8C4E5 !important; text-decoration: none; }
[data-testid="stSidebar"] hr { border-color: #2E5FA8; }
[data-testid="stMetric"] {
    background: #F2F5FA;
    border: 1px solid #DDE3EC;
    border-radius: 6px;
    padding: 0.8rem 1rem;
}
h1 { color: #1B3A6B !important; }
h2 { color: #1B3A6B !important; }
h3 { color: #2E5FA8 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### SAPDE")
    st.markdown("Sistema de Analítica Predictiva  \nde Deserción Estudiantil")
    st.divider()
    st.page_link("app.py",                            label="Inicio")
    st.page_link("pages/1_Prediccion.py",              label="Prediccion Individual")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",   label="Estudiantes en Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",        label="Metricas de Modelos")
    st.divider()
    st.markdown("Universidad de Pamplona  \nIngeniería de Sistemas · 2026")

try:
    pipeline, meta, feature_names = cargar_modelo()
    modelo_ok = True
except Exception as exc:
    modelo_ok = False
    st.error(f"Error al cargar el modelo: {exc}")

try:
    engine = obtener_engine()
    db_ok = True
except Exception:
    db_ok = False

st.title("SAPDE — Sistema de Prediccion de Desercion Estudiantil")
st.markdown(
    "Universidad de Pamplona — Facultad de Ciencias y Procesos del Conocimiento  \n"
    "Programa de Ingeniería de Sistemas"
)
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    if modelo_ok:
        st.metric("Modelo activo", f"{meta['nombre']} {meta['version']}", "En linea")
    else:
        st.metric("Modelo activo", "No disponible")
with col2:
    st.metric("AUC-ROC", f"{meta['auc_roc']:.4f}" if modelo_ok else "—",
              help="Area bajo la curva ROC en el conjunto de prueba")
with col3:
    st.metric("Recall", f"{meta['recall']:.4f}" if modelo_ok else "—",
              help="Sensibilidad del modelo para detectar desertores")
with col4:
    st.metric("Base de datos", "PostgreSQL" if db_ok else "SQLite (local)")

st.divider()

if db_ok:
    df_all   = cargar_predicciones_db(engine, umbral_proba=0.0)
    umbral_u = meta["umbral"] if modelo_ok else 0.5
    df_riesgo = cargar_predicciones_db(engine, umbral_proba=umbral_u)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicciones registradas", len(df_all))
    with c2:
        n = len(df_riesgo)
        pct = f"{n/max(len(df_all),1)*100:.1f}%" if len(df_all) > 0 else None
        st.metric("En riesgo detectados", n, delta=pct)
    with c3:
        muy_alto = int((df_all["nivel_riesgo"] == "muy_alto").sum()) if len(df_all) > 0 else 0
        st.metric("Riesgo muy alto", muy_alto)
    with c4:
        proba_m = f"{df_all['probabilidad'].mean():.3f}" if len(df_all) > 0 else "—"
        st.metric("Probabilidad media", proba_m)

    if len(df_all) > 0:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("Distribucion por nivel de riesgo")
            conteo = df_all["nivel_riesgo"].value_counts().reindex(
                ["muy_alto", "alto", "moderado", "bajo"], fill_value=0
            )
            fig_pie = go.Figure(go.Pie(
                labels=[NIVEL_ETIQUETA.get(n, n) for n in conteo.index],
                values=conteo.values,
                marker_colors=[NIVELES_COLOR[n] for n in conteo.index],
                hole=0.45,
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=13),
            ))
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_g2:
            st.subheader("Distribucion de probabilidad de desercion")
            fig_hist = go.Figure(go.Histogram(
                x=df_all["probabilidad"],
                nbinsx=20,
                marker_color="#2E5FA8",
                opacity=0.85,
            ))
            if modelo_ok:
                fig_hist.add_vline(
                    x=meta["umbral"],
                    line_dash="dash",
                    line_color="#C62828",
                    line_width=2,
                    annotation_text=f"Umbral ({meta['umbral']:.2f})",
                    annotation_position="top right",
                    annotation_font_color="#C62828",
                )
            fig_hist.update_layout(
                xaxis_title="P(desercion)",
                yaxis_title="Frecuencia",
                margin=dict(t=10, b=40, l=40, r=20),
                height=300,
                paper_bgcolor="white",
                plot_bgcolor="#FAFBFC",
            )
            fig_hist.update_xaxes(gridcolor="#EAEAEA")
            fig_hist.update_yaxes(gridcolor="#EAEAEA")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Ultimas predicciones registradas")
        df_show = df_all.head(10).copy()
        df_show["nivel_riesgo"] = df_show["nivel_riesgo"].map(
            lambda n: NIVEL_ETIQUETA.get(n, n)
        )
        st.dataframe(
            df_show.rename(columns={
                "student_id": "ID Estudiante",
                "semestre": "Semestre",
                "probabilidad": "P(desercion)",
                "nivel_riesgo": "Nivel de Riesgo",
                "modelo": "Modelo",
                "timestamp": "Fecha",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No hay predicciones registradas. Ingrese a la pagina Prediccion Individual para evaluar un estudiante.")
else:
    st.warning("Base de datos no disponible. Las metricas aparecen cuando se establece la conexion.")

st.divider()
st.subheader("Modelos entrenados disponibles")
artefactos = cargar_todos_artefactos()
if artefactos:
    import pandas as pd
    df_art = pd.DataFrame(artefactos)[["nombre", "version", "auc_roc", "recall", "f1_score", "umbral"]]
    df_art = df_art.rename(columns={
        "nombre": "Modelo", "version": "Version",
        "auc_roc": "AUC-ROC", "recall": "Recall",
        "f1_score": "F1-Score", "umbral": "Umbral",
    })
    df_art = df_art.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    st.dataframe(
        df_art.style
            .highlight_max(subset=["AUC-ROC", "Recall", "F1-Score"], color="#D5E8F5")
            .format({"AUC-ROC": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "Umbral": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No hay modelos entrenados. Ejecute: python -m sapde.models.train")
