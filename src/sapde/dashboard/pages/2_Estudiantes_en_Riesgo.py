"""
Pagina 2 — Estudiantes en riesgo de desercion.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from sapde.dashboard.utils import (
    NIVEL_ETIQUETA,
    NIVELES_COLOR,
    cargar_predicciones_db,
    obtener_engine,
)

st.set_page_config(page_title="SAPDE — Estudiantes en Riesgo", page_icon=None, layout="wide")

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
    st.markdown("Estudiantes en Riesgo")
    st.divider()
    st.page_link("app.py",                            label="Inicio")
    st.page_link("pages/1_Prediccion.py",              label="Prediccion Individual")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",   label="Estudiantes en Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",        label="Metricas de Modelos")
    st.divider()
    umbral_filtro = st.slider("Probabilidad minima", 0.0, 1.0, 0.30, 0.05,
                              help="Mostrar solo estudiantes con P(desercion) >= este valor")
    niveles_filtro = st.multiselect(
        "Niveles de riesgo",
        options=["muy_alto", "alto", "moderado", "bajo"],
        default=["muy_alto", "alto"],
        format_func=lambda n: NIVEL_ETIQUETA.get(n, n),
    )

st.title("Estudiantes en Riesgo de Desercion")
st.divider()

try:
    engine = obtener_engine()
except Exception as exc:
    st.error(f"Base de datos no disponible: {exc}")
    st.stop()

df = cargar_predicciones_db(engine, umbral_proba=umbral_filtro)

if df.empty:
    st.info("No hay predicciones que superen el umbral seleccionado. "
            "Ingrese a Prediccion Individual para evaluar estudiantes.")
    st.stop()

if niveles_filtro:
    df = df[df["nivel_riesgo"].isin(niveles_filtro)]

if df.empty:
    st.warning("Ningun estudiante coincide con los filtros seleccionados.")
    st.stop()

# Metricas rapidas
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total filtrados", len(df))
with c2:
    muy_alto = int((df["nivel_riesgo"] == "muy_alto").sum())
    st.metric("Riesgo muy alto", muy_alto)
with c3:
    alto = int((df["nivel_riesgo"] == "alto").sum())
    st.metric("Riesgo alto", alto)
with c4:
    proba_max = df["probabilidad"].max()
    st.metric("Mayor probabilidad", f"{proba_max:.2%}")

st.divider()

col_chart, col_pie = st.columns([3, 2])

with col_chart:
    st.subheader("Probabilidad por semestre y nivel de riesgo")
    fig_scatter = px.scatter(
        df,
        x="semestre",
        y="probabilidad",
        color="nivel_riesgo",
        color_discrete_map=NIVELES_COLOR,
        hover_data=["student_id", "modelo"],
        labels={
            "semestre":     "Semestre",
            "probabilidad": "P(desercion)",
            "nivel_riesgo": "Nivel",
        },
    )
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.8))
    fig_scatter.update_layout(
        height=320,
        margin=dict(t=20, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFC",
        legend_title_text="Nivel de riesgo",
    )
    fig_scatter.update_xaxes(gridcolor="#EAEAEA")
    fig_scatter.update_yaxes(gridcolor="#EAEAEA")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_pie:
    st.subheader("Distribucion por nivel")
    conteo = df["nivel_riesgo"].value_counts().reindex(
        ["muy_alto", "alto", "moderado", "bajo"], fill_value=0
    )
    conteo = conteo[conteo > 0]
    fig_pie = go.Figure(go.Pie(
        labels=[NIVEL_ETIQUETA.get(n, n) for n in conteo.index],
        values=conteo.values,
        marker_colors=[NIVELES_COLOR[n] for n in conteo.index],
        hole=0.42,
        textinfo="label+value",
        textfont=dict(size=13),
        hoverinfo="label+value+percent",
    ))
    fig_pie.update_layout(
        height=320,
        margin=dict(t=20, b=20),
        showlegend=False,
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Tabla de estudiantes
st.subheader("Lista de estudiantes")

busqueda = st.text_input("Filtrar por ID de estudiante", placeholder="EST_...")
if busqueda:
    df = df[df["student_id"].str.contains(busqueda, case=False, na=False)]

df_display = df.copy()
df_display["nivel_riesgo"] = df_display["nivel_riesgo"].map(
    lambda n: NIVEL_ETIQUETA.get(n, n)
)
df_display["probabilidad"] = df_display["probabilidad"].map(lambda p: f"{p:.2%}")

st.dataframe(
    df_display.rename(columns={
        "student_id":   "ID Estudiante",
        "semestre":     "Semestre",
        "probabilidad": "P(desercion)",
        "nivel_riesgo": "Nivel de riesgo",
        "modelo":       "Modelo",
        "timestamp":    "Ultima prediccion",
    }),
    use_container_width=True,
    hide_index=True,
    height=360,
)

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar lista como CSV",
    data=csv_data,
    file_name="estudiantes_en_riesgo.csv",
    mime="text/csv",
)

# Detalle de un estudiante
st.divider()
st.subheader("Explicacion SHAP almacenada de un estudiante")
ids_disponibles = df["student_id"].unique().tolist()
id_seleccionado = st.selectbox("Seleccionar estudiante:", options=ids_disponibles)

if id_seleccionado and st.button("Ver detalle"):
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
            color = NIVELES_COLOR.get(nivel, "#555")
            etiq  = NIVEL_ETIQUETA.get(nivel, nivel)
            st.markdown(f"**Nivel de riesgo:** {etiq}")
            st.markdown(f"**Probabilidad:** {reg.probabilidad:.2%}")
            st.markdown(f"**Semestre:** {reg.semestre}")
            st.markdown(f"**Modelo:** {reg.modelo_nombre} {reg.modelo_version}")
            st.markdown(f"**Fecha:** {reg.timestamp}")

            if shap_data:
                st.markdown("---")
                st.markdown("**Factores que elevan el riesgo:**")
                for f in shap_data.get("top_factores_riesgo", []):
                    st.markdown(f"- `{f}`")
                st.markdown("**Factores protectores:**")
                for f in shap_data.get("top_factores_protectores", []):
                    st.markdown(f"- `{f}`")

        with col_shap:
            if shap_data and shap_data.get("feature_contributions"):
                sv = shap_data["feature_contributions"]
                sorted_sv = sorted(sv.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
                nombres = [k for k, _ in sorted_sv]
                valores = [v for _, v in sorted_sv]
                colores = ["#C62828" if v > 0 else "#1B3A6B" for v in valores]

                fig_bar = go.Figure(go.Bar(
                    x=valores[::-1],
                    y=nombres[::-1],
                    orientation="h",
                    marker_color=colores[::-1],
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                fig_bar.add_vline(x=0, line_color="#555555", line_width=1)
                fig_bar.update_layout(
                    title=dict(text="Contribucion SHAP por variable", font=dict(size=13, color="#1A1A2E")),
                    xaxis_title="Valor SHAP",
                    xaxis=dict(gridcolor="#EAEAEA"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    height=380,
                    margin=dict(t=50, b=40, l=160, r=20),
                    paper_bgcolor="white",
                    plot_bgcolor="#FAFBFC",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No hay datos SHAP almacenados para este estudiante.")
    else:
        st.warning("No se encontro registro para el estudiante seleccionado.")
