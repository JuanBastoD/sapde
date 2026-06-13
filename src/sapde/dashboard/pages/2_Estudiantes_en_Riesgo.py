"""
Página 2 — Estudiantes en riesgo.

Lista todos los estudiantes con predicción de deserción almacenada,
permite filtrar por nivel de riesgo y ver el detalle de cada uno.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd

from sapde.dashboard.utils import (
    NIVEL_EMOJI,
    NIVELES_COLOR,
    cargar_predicciones_db,
    obtener_engine,
)

st.set_page_config(page_title="SAPDE — En Riesgo", page_icon="⚠️", layout="wide")

with st.sidebar:
    st.title("⚠️ Estudiantes en Riesgo")
    st.divider()
    st.page_link("app.py",                                  label="🏠 Inicio")
    st.page_link("pages/1_Prediccion.py",                   label="🔍 Predicción")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",        label="⚠️ En Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",             label="📊 Métricas")
    st.divider()
    umbral_filtro = st.slider("Umbral mínimo de probabilidad", 0.0, 1.0, 0.30, 0.05)
    niveles_filtro = st.multiselect(
        "Niveles de riesgo",
        options=["muy_alto", "alto", "moderado", "bajo"],
        default=["muy_alto", "alto"],
    )

st.title("⚠️ Estudiantes en Riesgo de Deserción")
st.divider()

try:
    engine = obtener_engine()
    db_ok = True
except Exception as exc:
    st.error(f"Base de datos no disponible: {exc}")
    st.stop()

df = cargar_predicciones_db(engine, umbral_proba=umbral_filtro)

if df.empty:
    st.info("No hay predicciones que superen el umbral seleccionado. "
            "Ve a **Predicción** para evaluar estudiantes.")
    st.stop()

# Aplicar filtro de nivel
if niveles_filtro:
    df = df[df["nivel_riesgo"].isin(niveles_filtro)]

if df.empty:
    st.warning("Ningún estudiante coincide con los filtros seleccionados.")
    st.stop()

# ── Métricas rápidas ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total filtrados", len(df))
with c2:
    muy_alto = (df["nivel_riesgo"] == "muy_alto").sum()
    st.metric(f"{NIVEL_EMOJI['muy_alto']} Muy alto", muy_alto)
with c3:
    alto = (df["nivel_riesgo"] == "alto").sum()
    st.metric(f"{NIVEL_EMOJI['alto']} Alto", alto)
with c4:
    proba_max = df["probabilidad"].max()
    st.metric("Mayor probabilidad", f"{proba_max:.2%}")

st.divider()

# ── Gráfica de dispersión semestre vs probabilidad ────────────────────────────
col_chart, col_pie = st.columns([3, 2])

with col_chart:
    st.subheader("Probabilidad por semestre y nivel de riesgo")
    color_map = {n: NIVELES_COLOR[n] for n in NIVELES_COLOR}
    fig_scatter = px.scatter(
        df,
        x="semestre",
        y="probabilidad",
        color="nivel_riesgo",
        color_discrete_map=color_map,
        hover_data=["student_id", "modelo"],
        labels={"semestre": "Semestre", "probabilidad": "P(deserción)", "nivel_riesgo": "Nivel"},
        size_max=10,
    )
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.8))
    fig_scatter.update_layout(height=320, margin=dict(t=20, b=40))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_pie:
    st.subheader("Por nivel de riesgo")
    conteo = df["nivel_riesgo"].value_counts().reindex(
        ["muy_alto", "alto", "moderado", "bajo"], fill_value=0
    )
    conteo = conteo[conteo > 0]
    fig_pie = go.Figure(go.Pie(
        labels=[f"{NIVEL_EMOJI[n]} {n.replace('_',' ')}" for n in conteo.index],
        values=conteo.values,
        marker_colors=[NIVELES_COLOR[n] for n in conteo.index],
        hole=0.4,
        textinfo="label+value",
    ))
    fig_pie.update_layout(height=320, margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Tabla de estudiantes ──────────────────────────────────────────────────────
st.subheader("Lista de estudiantes")

busqueda = st.text_input("Buscar por ID de estudiante", placeholder="EST_...")
if busqueda:
    df = df[df["student_id"].str.contains(busqueda, case=False, na=False)]

# Preparar tabla con emojis de nivel
df_display = df.copy()
df_display["nivel_riesgo"] = df_display["nivel_riesgo"].map(
    lambda n: f"{NIVEL_EMOJI.get(n, '')} {n.replace('_',' ').title()}"
)
df_display["probabilidad"] = df_display["probabilidad"].map(lambda p: f"{p:.2%}")

st.dataframe(
    df_display.rename(columns={
        "student_id":   "ID Estudiante",
        "semestre":     "Sem.",
        "probabilidad": "P(deserción)",
        "nivel_riesgo": "Nivel de riesgo",
        "modelo":       "Modelo",
        "timestamp":    "Última predicción",
    }),
    use_container_width=True,
    hide_index=True,
    height=350,
)

# ── Descarga CSV ──────────────────────────────────────────────────────────────
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar lista como CSV",
    data=csv_data,
    file_name="estudiantes_en_riesgo.csv",
    mime="text/csv",
)

# ── Detalle de un estudiante ──────────────────────────────────────────────────
st.divider()
st.subheader("Ver explicación de un estudiante")
ids_disponibles = df["student_id"].unique().tolist()
id_seleccionado = st.selectbox("Selecciona un estudiante:", options=ids_disponibles)

if id_seleccionado and st.button("Ver explicación SHAP almacenada"):
    from sqlalchemy.orm import sessionmaker
    from sapde.db.crud import obtener_ultima_prediccion

    Session = sessionmaker(bind=engine)
    db_sess = Session()
    try:
        reg = obtener_ultima_prediccion(db_sess, id_seleccionado)
    finally:
        db_sess.close()

    if reg:
        shap_data = reg.shap_dict()
        col_info, col_shap = st.columns([1, 2])
        with col_info:
            nivel = reg.nivel_riesgo
            st.markdown(f"**{NIVEL_EMOJI.get(nivel,'')} {nivel.replace('_',' ').upper()}**")
            st.markdown(f"- Probabilidad: **{reg.probabilidad:.2%}**")
            st.markdown(f"- Semestre: {reg.semestre}")
            st.markdown(f"- Modelo: {reg.modelo_nombre} {reg.modelo_version}")
            st.markdown(f"- Fecha: {reg.timestamp}")

            if shap_data:
                st.markdown("**Factores de riesgo:**")
                for f in shap_data.get("top_factores_riesgo", []):
                    st.markdown(f"  🔴 `{f}`")
                st.markdown("**Factores protectores:**")
                for f in shap_data.get("top_factores_protectores", []):
                    st.markdown(f"  🔵 `{f}`")

        with col_shap:
            if shap_data and shap_data.get("feature_contributions"):
                sv = shap_data["feature_contributions"]
                sorted_sv = sorted(sv.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
                nombres = [k for k, _ in sorted_sv]
                valores = [v for _, v in sorted_sv]
                colores = ["#e74c3c" if v > 0 else "#3498db" for v in valores]

                fig_bar = go.Figure(go.Bar(
                    x=valores[::-1],
                    y=nombres[::-1],
                    orientation="h",
                    marker_color=colores[::-1],
                ))
                fig_bar.add_vline(x=0, line_color="black", line_width=1)
                fig_bar.update_layout(
                    title="Contribución SHAP",
                    xaxis_title="Valor SHAP",
                    height=350,
                    margin=dict(t=40, b=40, l=150, r=20),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No hay datos SHAP almacenados para este estudiante.")
