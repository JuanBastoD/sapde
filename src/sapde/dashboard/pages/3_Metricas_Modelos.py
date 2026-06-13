"""
Pagina 3 — Metricas y comparativa de modelos entrenados.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sapde.dashboard.utils import cargar_shap_resumen, cargar_todos_artefactos

st.set_page_config(page_title="SAPDE — Metricas de Modelos", page_icon=None, layout="wide")

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
    st.markdown("Metricas de Modelos")
    st.divider()
    st.page_link("app.py",                            label="Inicio")
    st.page_link("pages/1_Prediccion.py",              label="Prediccion Individual")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",   label="Estudiantes en Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",        label="Metricas de Modelos")

st.title("Comparativa de Modelos Entrenados")
st.divider()

artefactos = cargar_todos_artefactos()
if not artefactos:
    st.info("No hay modelos entrenados. Ejecute: python -m sapde.models.train")
    st.stop()

df_art = pd.DataFrame(artefactos)
df_art = df_art.sort_values("auc_roc", ascending=False).reset_index(drop=True)
df_art["label"] = df_art["nombre"] + " v" + df_art["version"].astype(str)

# Tabla comparativa
st.subheader("Tabla de metricas")
df_show = df_art[["nombre", "version", "auc_roc", "recall", "f1_score", "umbral"]].copy()
df_show.columns = ["Modelo", "Version", "AUC-ROC", "Recall", "F1-Score", "Umbral"]
df_show.insert(0, "Mejor", "")
df_show.loc[df_show["AUC-ROC"] == df_show["AUC-ROC"].max(), "Mejor"] = "Mejor"

st.dataframe(
    df_show.style
        .highlight_max(subset=["AUC-ROC", "Recall", "F1-Score"], color="#D5E8F5")
        .format({
            "AUC-ROC":  "{:.4f}",
            "Recall":   "{:.4f}",
            "F1-Score": "{:.4f}",
            "Umbral":   "{:.4f}",
        }),
    use_container_width=True,
    hide_index=True,
)

mejor = df_art.iloc[0]
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Mejor modelo", mejor["nombre"])
with c2:
    st.metric("AUC-ROC", f"{mejor['auc_roc']:.4f}")
with c3:
    st.metric("Recall", f"{mejor['recall']:.4f}")
with c4:
    st.metric("F1-Score", f"{mejor['f1_score']:.4f}")

st.divider()

# Graficas comparativas
st.subheader("Comparativa visual")

metrica_sel = st.radio(
    "Metrica a visualizar",
    options=["auc_roc", "recall", "f1_score"],
    format_func=lambda m: {"auc_roc": "AUC-ROC", "recall": "Recall", "f1_score": "F1-Score"}[m],
    horizontal=True,
)

col_bar, col_radar = st.columns(2)

with col_bar:
    fig_bar = px.bar(
        df_art,
        x="label",
        y=metrica_sel,
        color=metrica_sel,
        color_continuous_scale=[[0, "#A8C4E5"], [0.5, "#2E5FA8"], [1, "#1B3A6B"]],
        range_color=[df_art[metrica_sel].min() * 0.95, 1.0],
        labels={"label": "Modelo", metrica_sel: metrica_sel.upper().replace("_", "-")},
        text_auto=".4f",
    )
    fig_bar.update_layout(
        title=dict(
            text={"auc_roc": "AUC-ROC", "recall": "Recall", "f1_score": "F1-Score"}[metrica_sel] + " por modelo",
            font=dict(size=13, color="#1A1A2E"),
        ),
        showlegend=False,
        coloraxis_showscale=False,
        height=350,
        margin=dict(t=50, b=60, l=40, r=20),
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFC",
    )
    fig_bar.update_xaxes(gridcolor="#EAEAEA")
    fig_bar.update_yaxes(gridcolor="#EAEAEA")
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_radar:
    categorias = ["AUC-ROC", "Recall", "F1-Score"]
    fig_radar = go.Figure()
    palette = ["#1B3A6B", "#C62828", "#2E7D32", "#E65100", "#6A0DAD"]

    for i, row in df_art.iterrows():
        valores = [row["auc_roc"], row["recall"], row["f1_score"]]
        fig_radar.add_trace(go.Scatterpolar(
            r=valores + [valores[0]],
            theta=categorias + [categorias[0]],
            name=row["label"],
            line=dict(color=palette[i % len(palette)], width=2),
            fill="toself",
            opacity=0.3,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(range=[0.5, 1.0], gridcolor="#DDE3EC"),
            angularaxis=dict(gridcolor="#DDE3EC"),
            bgcolor="#FAFBFC",
        ),
        title=dict(text="Radar de metricas", font=dict(size=13, color="#1A1A2E")),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=11)),
        height=380,
        margin=dict(t=50, b=80),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Importancia SHAP
st.divider()
st.subheader("Importancia de variables (SHAP)")

shap_resumen = cargar_shap_resumen()

if not shap_resumen:
    st.info("No hay reporte SHAP generado. Ejecute: python -m sapde.explain.run_shap")
else:
    todos_features: set[str] = set()
    for modelo_data in shap_resumen.values():
        todos_features.update(f["feature"] for f in modelo_data)

    modelos       = list(shap_resumen.keys())
    features_list = sorted(todos_features)

    matriz = pd.DataFrame(index=features_list, columns=modelos, dtype=float)
    for modelo, top_list in shap_resumen.items():
        for entry in top_list:
            matriz.loc[entry["feature"], modelo] = entry["importancia_media"]
    matriz = matriz.fillna(0.0)

    matriz_norm = matriz / (matriz.max() + 1e-9)
    matriz_norm["_mean"] = matriz_norm.mean(axis=1)
    matriz_norm = matriz_norm.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    matriz = matriz.loc[matriz_norm.index]

    top_n = st.slider("Numero de variables a mostrar", 5, 25, 15)
    matriz_norm    = matriz_norm.head(top_n)
    matriz_display = matriz.head(top_n)

    tab_heatmap, tab_barras = st.tabs(["Heatmap normalizado", "Importancia absoluta"])

    with tab_heatmap:
        fig_heat = px.imshow(
            matriz_norm.T,
            labels=dict(x="Variable", y="Modelo", color="Importancia norm."),
            color_continuous_scale=[[0, "#EEF2FA"], [0.5, "#6990CC"], [1, "#1B3A6B"]],
            aspect="auto",
            text_auto=".2f",
        )
        fig_heat.update_layout(
            height=max(300, 60 * len(modelos)),
            margin=dict(t=40, b=60, l=80, r=40),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_barras:
        modelo_sel = st.selectbox("Seleccionar modelo", options=modelos)
        col_data   = matriz_display[modelo_sel].sort_values(ascending=False)
        col_data   = col_data[col_data > 0]

        fig_imp = go.Figure(go.Bar(
            x=col_data.values[::-1],
            y=col_data.index[::-1],
            orientation="h",
            marker_color="#2E5FA8",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            title=dict(text=f"Importancia SHAP — {modelo_sel}", font=dict(size=13, color="#1A1A2E")),
            xaxis_title="|SHAP| promedio",
            xaxis=dict(gridcolor="#EAEAEA"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=max(300, 30 * len(col_data)),
            margin=dict(t=50, b=40, l=180, r=20),
            paper_bgcolor="white",
            plot_bgcolor="#FAFBFC",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Top 5 variables por modelo")
    cols = st.columns(len(modelos))
    for i, modelo in enumerate(modelos):
        top5 = [e["feature"] for e in shap_resumen[modelo][:5]]
        with cols[i]:
            st.markdown(f"**{modelo}**")
            for j, feat in enumerate(top5, 1):
                st.markdown(f"{j}. `{feat}`")
