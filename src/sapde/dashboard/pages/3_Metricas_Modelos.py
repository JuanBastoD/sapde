"""
Página 3 — Métricas y comparativa de modelos.

Muestra:
  - Tabla comparativa de todos los modelos entrenados
  - Gráficas de barras: AUC-ROC, Recall, F1
  - Top features por modelo (desde shap_resumen.json)
  - Heatmap de importancia SHAP
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sapde.dashboard.utils import cargar_shap_resumen, cargar_todos_artefactos

st.set_page_config(page_title="SAPDE — Métricas", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("📊 Métricas de Modelos")
    st.divider()
    st.page_link("app.py",                                  label="🏠 Inicio")
    st.page_link("pages/1_Prediccion.py",                   label="🔍 Predicción")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",        label="⚠️ En Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",             label="📊 Métricas")

st.title("📊 Comparativa de Modelos Entrenados")
st.divider()

# ── Artefactos ────────────────────────────────────────────────────────────────
artefactos = cargar_todos_artefactos()
if not artefactos:
    st.info("No hay modelos entrenados. Ejecuta `python -m sapde.models.train` primero.")
    st.stop()

df_art = pd.DataFrame(artefactos)
df_art = df_art.sort_values("auc_roc", ascending=False).reset_index(drop=True)
df_art["label"] = df_art["nombre"] + " v" + df_art["version"].astype(str)

# ── Tabla comparativa ─────────────────────────────────────────────────────────
st.subheader("Tabla de métricas")
df_show = df_art[["nombre", "version", "auc_roc", "recall", "f1_score", "umbral"]].copy()
df_show.columns = ["Modelo", "Versión", "AUC-ROC", "Recall", "F1-Score", "Umbral"]
df_show["Mejor"] = ""
df_show.loc[df_show["AUC-ROC"] == df_show["AUC-ROC"].max(), "Mejor"] = "⭐"

st.dataframe(
    df_show.style
        .highlight_max(subset=["AUC-ROC", "Recall", "F1-Score"], color="#d5f5e3")
        .format({
            "AUC-ROC":  "{:.4f}",
            "Recall":   "{:.4f}",
            "F1-Score": "{:.4f}",
            "Umbral":   "{:.4f}",
        }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Gráficas de barras comparativas ──────────────────────────────────────────
st.subheader("Comparativa visual")

metrica_sel = st.radio(
    "Métrica a visualizar",
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
        color_continuous_scale="RdYlGn",
        range_color=[df_art[metrica_sel].min() * 0.95, 1.0],
        labels={"label": "Modelo", metrica_sel: metrica_sel.upper().replace("_","-")},
        text_auto=".4f",
    )
    fig_bar.update_layout(
        title=f"{metrica_sel.upper().replace('_','-')} por modelo",
        showlegend=False,
        height=350,
        margin=dict(t=50, b=60, l=40, r=20),
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_radar:
    # Radar chart (las 3 métricas normalizadas)
    categorias = ["AUC-ROC", "Recall", "F1-Score"]
    fig_radar = go.Figure()
    colores_modelo = px.colors.qualitative.Set2

    for i, row in df_art.iterrows():
        valores = [row["auc_roc"], row["recall"], row["f1_score"]]
        fig_radar.add_trace(go.Scatterpolar(
            r=valores + [valores[0]],
            theta=categorias + [categorias[0]],
            name=row["label"],
            line=dict(color=colores_modelo[i % len(colores_modelo)]),
            fill="toself",
            opacity=0.4,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0.5, 1.0])),
        title="Radar de métricas",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        height=380,
        margin=dict(t=50, b=80),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── SHAP feature importance ───────────────────────────────────────────────────
st.divider()
st.subheader("Importancia de features (SHAP)")

shap_resumen = cargar_shap_resumen()

if not shap_resumen:
    st.info("No hay reporte SHAP generado. Ejecuta `python -m sapde.explain.run_shap` primero.")
else:
    # Construir matriz de importancia para heatmap
    todos_features: set[str] = set()
    for modelo_data in shap_resumen.values():
        todos_features.update(f["feature"] for f in modelo_data)

    modelos = list(shap_resumen.keys())
    features_sorted = sorted(todos_features)

    matriz = pd.DataFrame(index=features_sorted, columns=modelos, dtype=float)
    for modelo, top_list in shap_resumen.items():
        for entry in top_list:
            f = entry["feature"]
            v = entry["importancia_media"]
            matriz.loc[f, modelo] = v
    matriz = matriz.fillna(0.0)

    # Normalizar por columna (máx. por modelo = 1)
    matriz_norm = matriz / (matriz.max() + 1e-9)

    # Ordenar features por importancia promedio
    matriz_norm["mean"] = matriz_norm.mean(axis=1)
    matriz_norm = matriz_norm.sort_values("mean", ascending=False).drop(columns=["mean"])
    matriz = matriz.loc[matriz_norm.index]

    top_n = st.slider("Número de features a mostrar", 5, 25, 15)
    matriz_norm = matriz_norm.head(top_n)
    matriz_display = matriz.head(top_n)

    tab_heatmap, tab_barras = st.tabs(["Heatmap normalizado", "Importancia absoluta"])

    with tab_heatmap:
        fig_heat = px.imshow(
            matriz_norm.T,
            labels=dict(x="Feature", y="Modelo", color="Importancia norm."),
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=".2f",
        )
        fig_heat.update_layout(
            height=max(300, 60 * len(modelos)),
            margin=dict(t=40, b=60, l=80, r=40),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_barras:
        modelo_sel = st.selectbox("Seleccionar modelo", options=modelos)
        col_data = matriz_display[modelo_sel].sort_values(ascending=False)
        col_data = col_data[col_data > 0]

        fig_imp = go.Figure(go.Bar(
            x=col_data.values[::-1],
            y=col_data.index[::-1],
            orientation="h",
            marker_color="#3498db",
        ))
        fig_imp.update_layout(
            title=f"Importancia SHAP — {modelo_sel}",
            xaxis_title="|SHAP| medio",
            height=max(300, 30 * len(col_data)),
            margin=dict(t=50, b=40, l=180, r=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Top 5 features por modelo
    st.markdown("#### Top 5 features por modelo")
    cols = st.columns(len(modelos))
    for i, modelo in enumerate(modelos):
        top5 = [e["feature"] for e in shap_resumen[modelo][:5]]
        with cols[i]:
            st.markdown(f"**{modelo}**")
            for j, feat in enumerate(top5, 1):
                st.markdown(f"{j}. `{feat}`")
