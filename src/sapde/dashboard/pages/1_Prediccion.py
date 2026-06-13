"""
Página 1 — Predicción individual de deserción.

Permite ingresar los datos de un estudiante y obtiene:
  - Probabilidad de deserción
  - Nivel de riesgo con gauge
  - Explicación SHAP (waterfall interactivo)
  - Recomendaciones de intervención
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sapde.dashboard.utils import (
    NIVEL_EMOJI,
    NIVELES_COLOR,
    cargar_explainer,
    cargar_modelo,
    entrada_a_features,
    explicar,
    guardar_prediccion_db,
    nivel_riesgo,
    obtener_engine,
    predecir,
)

st.set_page_config(page_title="SAPDE — Predicción", page_icon="🔍", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Predicción")
    st.caption("Ingresa los datos académicos del estudiante para predecir el riesgo de deserción.")
    st.divider()
    st.page_link("app.py",                                  label="🏠 Inicio")
    st.page_link("pages/1_Prediccion.py",                   label="🔍 Predicción")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",        label="⚠️ En Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",             label="📊 Métricas")
    st.divider()
    incluir_shap = st.checkbox("Calcular explicación SHAP", value=True,
                               help="Más lento para SVM/MLP (~5-15 s)")
    guardar_db = st.checkbox("Guardar en base de datos", value=True)

# ── Carga de recursos ─────────────────────────────────────────────────────────
try:
    pipeline, meta, feature_names = cargar_modelo()
except Exception as exc:
    st.error(f"Modelo no disponible: {exc}")
    st.stop()

try:
    engine = obtener_engine()
    db_disponible = True
except Exception:
    db_disponible = False

st.title("🔍 Predicción de Riesgo de Deserción")
st.markdown(f"Modelo activo: **{meta['nombre']} {meta['version']}** · AUC = {meta['auc_roc']:.4f} · Umbral = {meta['umbral']:.4f}")
st.divider()

# ── Formulario de entrada ─────────────────────────────────────────────────────
with st.form("formulario_prediccion"):
    st.subheader("Datos del estudiante")

    col_id, col_sem = st.columns([2, 1])
    with col_id:
        student_id = st.text_input("ID del estudiante", value="EST_001", max_chars=50)
    with col_sem:
        semestre = st.number_input("Semestre", min_value=1, max_value=12, value=3, step=1)

    st.markdown("#### Rendimiento académico")
    c1, c2, c3 = st.columns(3)
    with c1:
        promedio_semestre = st.number_input("Promedio semestre", 0.0, 5.0, 3.2, 0.1, format="%.2f")
        creditos_matriculados = st.number_input("Créditos matriculados", 0, 30, 18, 1)
        num_asignaturas_reprobadas = st.number_input("Asignaturas reprobadas", 0, 10, 1, 1)
    with c2:
        promedio_acumulado = st.number_input("Promedio acumulado", 0.0, 5.0, 3.4, 0.1, format="%.2f")
        creditos_aprobados = st.number_input("Créditos aprobados", 0, 30, 15, 1)
        tasa_reprobacion = st.number_input("Tasa reprobación", 0.0, 1.0, 0.1, 0.01, format="%.2f")
    with c3:
        tendencia_calificaciones = st.number_input("Tendencia notas", -2.0, 2.0, -0.2, 0.1, format="%.2f",
                                                    help="promedio_semestre − promedio_acumulado")
        porcentaje_avance_real = st.number_input("% Avance real", 0.0, 100.0, 28.0, 0.5, format="%.1f")
        semestres_cursados = st.number_input("Semestres cursados", 0, 20, 3, 1)

    st.markdown("#### Historial y cohorte")
    c4, c5 = st.columns(2)
    with c4:
        interrupciones_previas = st.number_input("Interrupciones previas", 0, 10, 0, 1)
        anio_cohorte = st.number_input("Año de ingreso", 2010, 2030, 2022, 1)
    with c5:
        semestre_cohorte = st.selectbox("Semestre de ingreso", [1, 2], index=0)
        score_riesgo_bruto = st.number_input("Score riesgo ETL", 0.0, 1.0, 0.35, 0.01, format="%.3f",
                                              help="Calculado por el ETL C++")

    submitted = st.form_submit_button("🔮 Predecir riesgo de deserción", use_container_width=True)

# ── Procesamiento ─────────────────────────────────────────────────────────────
if submitted:
    max_aprob = creditos_matriculados
    if creditos_aprobados > max_aprob:
        st.error(f"Créditos aprobados ({creditos_aprobados}) no puede superar los matriculados ({max_aprob}).")
        st.stop()

    ratio_aprobacion = creditos_aprobados / max(creditos_matriculados, 1)
    creditos_esperados = semestre * 18
    deficit = max(0.0, float(creditos_esperados - creditos_aprobados))

    datos_entrada = {
        "semestre":                   semestre,
        "promedio_semestre":          promedio_semestre,
        "promedio_acumulado":         promedio_acumulado,
        "tendencia_calificaciones":   tendencia_calificaciones,
        "creditos_matriculados":      creditos_matriculados,
        "creditos_aprobados":         creditos_aprobados,
        "porcentaje_avance_real":     porcentaje_avance_real,
        "num_asignaturas_reprobadas": num_asignaturas_reprobadas,
        "tasa_reprobacion":           tasa_reprobacion,
        "semestres_cursados":         semestres_cursados,
        "interrupciones_previas":     interrupciones_previas,
        "ratio_aprobacion":           ratio_aprobacion,
        "deficit_creditos":           deficit,
        "score_riesgo_bruto":         score_riesgo_bruto,
        "anio_cohorte":               anio_cohorte,
        "semestre_cohorte":           semestre_cohorte,
    }

    with st.spinner("Calculando predicción..."):
        resultado = predecir(datos_entrada, pipeline, meta, feature_names)

    proba = resultado["probabilidad"]
    nivel = resultado["nivel_riesgo"]
    color = NIVELES_COLOR[nivel]
    emoji = NIVEL_EMOJI[nivel]

    # ── Panel de resultado ────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Resultado para: **{student_id}** — Semestre {semestre}")

    col_gauge, col_detail = st.columns([1, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Probabilidad de deserción (%)"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 20],   "color": "#d5f5e3"},
                    {"range": [20, meta["umbral"] * 100], "color": "#fef9e7"},
                    {"range": [meta["umbral"] * 100, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "#e74c3c", "width": 3},
                    "thickness": 0.85,
                    "value": meta["umbral"] * 100,
                },
            },
            number={"suffix": "%", "font": {"size": 28}},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_detail:
        st.markdown(f"### {emoji} Nivel de riesgo: **{nivel.replace('_',' ').upper()}**")
        st.markdown(f"- Probabilidad: **{proba:.2%}**")
        st.markdown(f"- Predicción: {'⚠️ Deserta' if resultado['prediccion'] else '✅ Permanece'}")
        st.markdown(f"- Umbral de clasificación: **{meta['umbral']:.4f}**")
        st.markdown(f"- Modelo: {meta['nombre']} {meta['version']}")

        st.markdown("---")
        recom_map = {
            "muy_alto": [
                "🚨 Iniciar acompañamiento académico urgente",
                "📅 Citar al consejero en los próximos 5 días",
                "📋 Revisar posibilidad de matrícula diferencial",
            ],
            "alto": [
                "📚 Inscribir en programa de tutoría entre pares",
                "📊 Monitoreo mensual de avance académico",
            ],
            "moderado": [
                "📧 Enviar correo de seguimiento con recursos de apoyo",
                "👥 Alertar al monitor de grupo",
            ],
            "bajo": ["✅ Mantener seguimiento regular en el sistema"],
        }
        st.markdown("**Recomendaciones:**")
        for r in recom_map.get(nivel, []):
            st.markdown(f"  {r}")

    # ── Guardar en DB ─────────────────────────────────────────────────────────
    if guardar_db and db_disponible:
        try:
            guardar_prediccion_db(engine, student_id, semestre, resultado, datos_entrada)
            st.success("Predicción guardada en la base de datos.")
        except Exception as exc:
            st.warning(f"No se pudo guardar en DB: {exc}")

    # ── Explicación SHAP ──────────────────────────────────────────────────────
    if incluir_shap:
        st.divider()
        st.subheader("Explicación SHAP — ¿Por qué esta predicción?")

        with st.spinner("Calculando SHAP... (puede tardar ~10 s para SVM/MLP)"):
            try:
                explainer = cargar_explainer(pipeline, feature_names)
                X_raw = resultado["X"]
                shap_exp = explicar(X_raw, explainer)
            except Exception as exc:
                st.error(f"SHAP no disponible: {exc}")
                shap_exp = None

        if shap_exp:
            sv = shap_exp["shap_values"]
            fn = shap_exp["feature_names"]
            base_val = shap_exp["base_value"]

            # Ordenar por |SHAP|
            arr = np.array(sv)
            sorted_idx = np.argsort(np.abs(arr))[::-1][:15]
            names_sorted = [fn[i] for i in sorted_idx]
            vals_sorted  = [arr[i] for i in sorted_idx]
            colors_sorted = ["#e74c3c" if v > 0 else "#3498db" for v in vals_sorted]

            fig_shap = go.Figure(go.Bar(
                x=vals_sorted[::-1],
                y=names_sorted[::-1],
                orientation="h",
                marker_color=colors_sorted[::-1],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_shap.add_vline(x=0, line_color="black", line_width=1)
            fig_shap.update_layout(
                title=f"Contribución SHAP por feature (base = {base_val:.4f})",
                xaxis_title="Valor SHAP (positivo → mayor riesgo)",
                height=500,
                margin=dict(t=50, b=40, l=160, r=40),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            col_r, col_p = st.columns(2)
            with col_r:
                st.markdown("**Factores de riesgo** (aumentan probabilidad):")
                positivos = [(fn[i], arr[i]) for i in sorted_idx if arr[i] > 0][:5]
                for nombre, val in positivos:
                    st.markdown(f"  - `{nombre}`: +{val:.4f}")
            with col_p:
                st.markdown("**Factores protectores** (reducen probabilidad):")
                negativos = [(fn[i], arr[i]) for i in sorted_idx if arr[i] < 0][:5]
                for nombre, val in negativos:
                    st.markdown(f"  - `{nombre}`: {val:.4f}")

            if guardar_db and db_disponible:
                try:
                    guardar_prediccion_db(engine, student_id, semestre, resultado,
                                          datos_entrada, shap_exp)
                except Exception:
                    pass
