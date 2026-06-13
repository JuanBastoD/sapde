"""
Pagina 1 — Prediccion individual de riesgo de desercion.
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
    NIVEL_ETIQUETA,
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

st.set_page_config(page_title="SAPDE — Prediccion", page_icon=None, layout="wide")

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
    st.markdown("Prediccion Individual")
    st.divider()
    st.page_link("app.py",                            label="Inicio")
    st.page_link("pages/1_Prediccion.py",              label="Prediccion Individual")
    st.page_link("pages/2_Estudiantes_en_Riesgo.py",   label="Estudiantes en Riesgo")
    st.page_link("pages/3_Metricas_Modelos.py",        label="Metricas de Modelos")
    st.divider()
    incluir_shap = st.checkbox("Calcular explicacion SHAP", value=True,
                               help="Tiempo adicional: 5-15 s para SVM y MLP")
    guardar_db   = st.checkbox("Guardar en base de datos", value=True)

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

st.title("Prediccion de Riesgo de Desercion")
st.markdown(
    f"Modelo activo: **{meta['nombre']} {meta['version']}** "
    f"— AUC-ROC: **{meta['auc_roc']:.4f}** "
    f"— Umbral de clasificacion: **{meta['umbral']:.4f}**"
)
st.divider()

with st.form("formulario_prediccion"):
    st.subheader("Datos del estudiante")

    col_id, col_sem = st.columns([2, 1])
    with col_id:
        student_id = st.text_input("Identificador del estudiante", value="EST_001", max_chars=50)
    with col_sem:
        semestre = st.number_input("Semestre cursado", min_value=1, max_value=12, value=6, step=1)

    st.markdown("**Rendimiento academico del semestre**")
    c1, c2, c3 = st.columns(3)
    with c1:
        promedio_semestre    = st.number_input("Promedio del semestre", 0.0, 5.0, 2.54, 0.01, format="%.2f")
        creditos_matriculados = st.number_input("Creditos matriculados", 0, 30, 21, 1)
        num_asignaturas_reprobadas = st.number_input("Asignaturas reprobadas", 0, 15, 2, 1)
    with c2:
        promedio_acumulado   = st.number_input("Promedio acumulado", 0.0, 5.0, 3.19, 0.01, format="%.2f")
        creditos_aprobados   = st.number_input("Creditos aprobados", 0, 30, 14, 1)
        tasa_reprobacion     = st.number_input("Tasa de reprobacion (0-1)", 0.0, 1.0, 0.43, 0.01, format="%.2f")
    with c3:
        tendencia_calificaciones = st.number_input(
            "Tendencia de calificaciones", -2.0, 2.0, -0.23, 0.01, format="%.2f",
            help="Diferencia entre promedio de este semestre y el acumulado"
        )
        porcentaje_avance_real = st.number_input("Porcentaje de avance del programa (%)", 0.0, 100.0, 41.98, 0.1, format="%.2f")
        semestres_cursados   = st.number_input("Semestres cursados hasta la fecha", 0, 20, 6, 1)

    st.markdown("**Historial y cohorte**")
    c4, c5 = st.columns(2)
    with c4:
        interrupciones_previas = st.number_input("Interrupciones previas de matricula", 0, 10, 0, 1)
        anio_cohorte           = st.number_input("Año de ingreso al programa", 2010, 2030, 2023, 1)
    with c5:
        semestre_cohorte = st.selectbox("Semestre de ingreso", [1, 2], index=1,
                                        format_func=lambda x: f"Semestre {x}")

    submitted = st.form_submit_button("Calcular prediccion de riesgo", use_container_width=True)

if submitted:
    if creditos_aprobados > creditos_matriculados:
        st.error(f"Los creditos aprobados ({creditos_aprobados}) no pueden superar los matriculados ({creditos_matriculados}).")
        st.stop()

    # Calcular derivados con la misma formula del ETL (fallback.py _transformar_chunk)
    ratio_aprobacion   = creditos_aprobados / max(creditos_matriculados, 1)
    deficit            = float(creditos_matriculados - creditos_aprobados)
    f_nota             = (5.0 - promedio_acumulado) / 4.0
    f_repro            = tasa_reprobacion
    f_avance           = 1.0 - porcentaje_avance_real / 100.0
    f_interr           = min(interrupciones_previas / 2.0, 1.0)
    score_riesgo_bruto = 0.40 * f_nota + 0.30 * f_repro + 0.20 * f_avance + 0.10 * f_interr

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

    with st.spinner("Calculando prediccion..."):
        resultado = predecir(datos_entrada, pipeline, meta, feature_names)

    proba  = resultado["probabilidad"]
    nivel  = resultado["nivel_riesgo"]
    color  = NIVELES_COLOR[nivel]
    etiq   = NIVEL_ETIQUETA[nivel]

    st.divider()
    st.subheader(f"Resultado — Estudiante {student_id} | Semestre {semestre}")

    col_gauge, col_detail = st.columns([1, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            title={"text": "Probabilidad de desercion (%)", "font": {"size": 15, "color": "#1A1A2E"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "bordercolor": "#DDE3EC",
                "steps": [
                    {"range": [0, 20],                        "color": "#E8F5E9"},
                    {"range": [20, meta["umbral"] * 100],     "color": "#FFF9C4"},
                    {"range": [meta["umbral"] * 100, 70],     "color": "#FFE0B2"},
                    {"range": [70, 100],                      "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "#C62828", "width": 2},
                    "thickness": 0.8,
                    "value": meta["umbral"] * 100,
                },
            },
            number={"suffix": "%", "font": {"size": 32, "color": color}},
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(t=50, b=10, l=30, r=30),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_detail:
        st.markdown(f"**Nivel de riesgo:** {etiq}")
        st.markdown(f"**Probabilidad estimada:** {proba:.2%}")
        st.markdown(f"**Clasificacion:** {'Desertor probable' if resultado['prediccion'] else 'Permanece'}")
        st.markdown(f"**Umbral de clasificacion:** {meta['umbral']:.4f}")
        st.markdown(f"**Modelo utilizado:** {meta['nombre']} {meta['version']}")

        recom_map = {
            "muy_alto": [
                "Iniciar acompañamiento academico de forma urgente",
                "Citar al estudiante con el consejero en los proximos 5 dias",
                "Evaluar posibilidad de matricula diferencial o alivio de carga",
            ],
            "alto": [
                "Inscribir en programa de tutoria entre pares",
                "Establecer monitoreo mensual de avance academico",
            ],
            "moderado": [
                "Enviar comunicacion de seguimiento con recursos de apoyo disponibles",
                "Alertar al monitor del grupo de trabajo",
            ],
            "bajo": [
                "Mantener seguimiento periodico regular en el sistema",
            ],
        }
        st.markdown("---")
        st.markdown("**Recomendaciones de intervencion:**")
        for rec in recom_map.get(nivel, []):
            st.markdown(f"- {rec}")

    if guardar_db and db_disponible:
        try:
            guardar_prediccion_db(engine, student_id, semestre, resultado, datos_entrada)
            st.success("Prediccion registrada en la base de datos.")
        except Exception as exc:
            st.warning(f"No se pudo registrar en la base de datos: {exc}")

    if incluir_shap:
        st.divider()
        st.subheader("Explicacion SHAP — Factores que determinan esta prediccion")

        with st.spinner("Calculando valores SHAP..."):
            try:
                explainer  = cargar_explainer(pipeline, feature_names)
                X_raw      = resultado["X"]
                shap_exp   = explicar(X_raw, explainer)
            except Exception as exc:
                st.error(f"Explicacion SHAP no disponible: {exc}")
                shap_exp = None

        if shap_exp:
            sv       = np.array(shap_exp["shap_values"])
            fn       = shap_exp["feature_names"]
            base_val = shap_exp["base_value"]

            sorted_idx    = np.argsort(np.abs(sv))[::-1][:15]
            names_sorted  = [fn[i] for i in sorted_idx]
            vals_sorted   = [float(sv[i]) for i in sorted_idx]
            colors_sorted = ["#C62828" if v > 0 else "#1B3A6B" for v in vals_sorted]

            fig_shap = go.Figure(go.Bar(
                x=vals_sorted[::-1],
                y=names_sorted[::-1],
                orientation="h",
                marker_color=colors_sorted[::-1],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_shap.add_vline(x=0, line_color="#555555", line_width=1)
            fig_shap.update_layout(
                title=dict(
                    text=f"Contribucion de cada variable (valor base = {base_val:.4f})",
                    font=dict(size=13, color="#1A1A2E"),
                ),
                xaxis_title="Valor SHAP — positivo aumenta el riesgo, negativo lo reduce",
                xaxis=dict(gridcolor="#EAEAEA"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                height=480,
                margin=dict(t=50, b=50, l=180, r=40),
                paper_bgcolor="white",
                plot_bgcolor="#FAFBFC",
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            col_r, col_p = st.columns(2)
            with col_r:
                st.markdown("**Factores que elevan el riesgo** (valores SHAP positivos):")
                for nombre, val in [(fn[i], float(sv[i])) for i in sorted_idx if sv[i] > 0][:5]:
                    st.markdown(f"- `{nombre}`: +{val:.4f}")
            with col_p:
                st.markdown("**Factores que reducen el riesgo** (valores SHAP negativos):")
                for nombre, val in [(fn[i], float(sv[i])) for i in sorted_idx if sv[i] < 0][:5]:
                    st.markdown(f"- `{nombre}`: {val:.4f}")

            if guardar_db and db_disponible:
                try:
                    guardar_prediccion_db(engine, student_id, semestre, resultado, datos_entrada, shap_exp)
                except Exception:
                    pass
