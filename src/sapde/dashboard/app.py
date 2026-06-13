"""
SAPDE Dashboard — Página principal (Inicio / Resumen del sistema).

Ejecutar:
    streamlit run src/sapde/dashboard/app.py
"""

import sys
from pathlib import Path

# Asegurar que src/ esté en el path (necesario cuando se lanza desde la raíz)
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import plotly.graph_objects as go
import streamlit as st

from sapde.dashboard.utils import (
    NIVEL_EMOJI,
    NIVELES_COLOR,
    cargar_modelo,
    cargar_predicciones_db,
    cargar_todos_artefactos,
    obtener_engine,
)

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SAPDE — Sistema de Predicción de Deserción",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 SAPDE")
    st.caption("Sistema de Analítica Predictiva\nde Deserción Estudiantil")
    st.divider()
    st.markdown("**Navegación:**")
    st.page_link("app.py",                                  label="🏠 Inicio")
    st.page_link("pages/1_Prediccion.py",                   label="🔍 Predicción")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",        label="⚠️ En Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",             label="📊 Métricas")
    st.divider()
    st.caption("Universidad de Pamplona\nIngeniería de Sistemas · 2026")

# ── Carga de recursos ─────────────────────────────────────────────────────────
try:
    pipeline, meta, feature_names = cargar_modelo()
    modelo_ok = True
except Exception as exc:
    modelo_ok = False
    st.error(f"Error cargando modelo: {exc}")

try:
    engine = obtener_engine()
    db_ok = True
except Exception:
    db_ok = False

# ── Encabezado ────────────────────────────────────────────────────────────────
st.title("🎓 SAPDE — Sistema de Predicción de Deserción")
st.markdown(
    "**Sistema de Analítica Predictiva de Deserción Estudiantil** | "
    "Universidad de Pamplona — Ingeniería de Sistemas"
)
st.divider()

# ── Estado del sistema ────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    if modelo_ok:
        st.metric("Modelo activo", f"{meta['nombre']} {meta['version']}", "Cargado ✓")
    else:
        st.metric("Modelo activo", "No disponible", delta_color="off")

with col2:
    if modelo_ok:
        st.metric("AUC-ROC", f"{meta['auc_roc']:.3f}", help="Área bajo la curva ROC")
    else:
        st.metric("AUC-ROC", "—")

with col3:
    if modelo_ok:
        st.metric("Recall", f"{meta['recall']:.3f}", help="Sensibilidad para clase positiva")
    else:
        st.metric("Recall", "—")

with col4:
    st.metric("Base de datos", "PostgreSQL ✓" if db_ok else "SQLite (offline)", "")

st.divider()

# ── Métricas de predicciones almacenadas ─────────────────────────────────────
if db_ok:
    df_all = cargar_predicciones_db(engine, umbral_proba=0.0)
    df_riesgo = cargar_predicciones_db(engine, umbral_proba=meta["umbral"] if modelo_ok else 0.5)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Predicciones totales", len(df_all))
    with col_b:
        n_riesgo = len(df_riesgo)
        st.metric("En riesgo detectados", n_riesgo,
                  delta=f"{n_riesgo/max(len(df_all),1)*100:.1f}%" if len(df_all) > 0 else None)
    with col_c:
        if len(df_all) > 0:
            muy_alto = (df_all["nivel_riesgo"] == "muy_alto").sum()
            st.metric("Riesgo muy alto", muy_alto)
        else:
            st.metric("Riesgo muy alto", 0)
    with col_d:
        if len(df_all) > 0:
            proba_media = df_all["probabilidad"].mean()
            st.metric("Probabilidad media", f"{proba_media:.3f}")
        else:
            st.metric("Probabilidad media", "—")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    if len(df_all) > 0:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("Distribución por nivel de riesgo")
            conteo = df_all["nivel_riesgo"].value_counts().reindex(
                ["muy_alto", "alto", "moderado", "bajo"], fill_value=0
            )
            colores = [NIVELES_COLOR[n] for n in conteo.index]
            etiquetas = [f"{NIVEL_EMOJI[n]} {n.replace('_', ' ').title()}" for n in conteo.index]

            fig_pie = go.Figure(go.Pie(
                labels=etiquetas,
                values=conteo.values,
                marker_colors=colores,
                hole=0.4,
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            ))
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_g2:
            st.subheader("Distribución de probabilidad")
            fig_hist = go.Figure(go.Histogram(
                x=df_all["probabilidad"],
                nbinsx=20,
                marker_color="#3498db",
                opacity=0.8,
            ))
            if modelo_ok:
                fig_hist.add_vline(
                    x=meta["umbral"], line_dash="dash",
                    line_color="#e74c3c",
                    annotation_text=f"Umbral ({meta['umbral']:.2f})",
                    annotation_position="top right",
                )
            fig_hist.update_layout(
                xaxis_title="P(deserción)",
                yaxis_title="Frecuencia",
                margin=dict(t=20, b=40, l=40, r=20),
                height=320,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Tabla de últimas predicciones
        st.subheader("Últimas predicciones registradas")
        df_show = df_all.head(10).copy()
        df_show["nivel_riesgo"] = df_show["nivel_riesgo"].map(
            lambda n: f"{NIVEL_EMOJI.get(n,'')} {n}"
        )
        st.dataframe(
            df_show.rename(columns={
                "student_id": "ID Estudiante",
                "semestre": "Sem.",
                "probabilidad": "P(deserción)",
                "nivel_riesgo": "Nivel de riesgo",
                "modelo": "Modelo",
                "timestamp": "Fecha",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No hay predicciones registradas aún. "
            "Ve a la página **Predicción** para evaluar un estudiante."
        )
else:
    st.warning("Base de datos no disponible. Las métricas se cargarán cuando se conecte.")

# ── Comparativa de modelos disponibles ───────────────────────────────────────
st.divider()
st.subheader("Modelos entrenados disponibles")
artefactos = cargar_todos_artefactos()
if artefactos:
    import pandas as pd
    df_art = pd.DataFrame(artefactos)[["nombre", "version", "auc_roc", "recall", "f1_score", "umbral"]]
    df_art = df_art.rename(columns={
        "nombre": "Modelo", "version": "Versión",
        "auc_roc": "AUC-ROC", "recall": "Recall",
        "f1_score": "F1-Score", "umbral": "Umbral"
    })
    df_art = df_art.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    st.dataframe(
        df_art.style.highlight_max(subset=["AUC-ROC", "Recall", "F1-Score"], color="#d5f5e3")
                    .format({"AUC-ROC": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "Umbral": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No hay modelos entrenados. Ejecuta `python -m sapde.models.train` primero.")
