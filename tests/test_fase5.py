"""
Tests Fase 5 — Dashboard Streamlit.

Los tests cubren la lógica de negocio del dashboard (utils.py) y
verifican que las páginas tienen sintaxis válida y las funciones clave
funcionan correctamente, sin necesitar un servidor Streamlit en ejecución.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Asegurar que src/ esté en el path
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sapde.dashboard.utils import (
    FEATURE_NAMES,
    NIVEL_EMOJI,
    NIVELES_COLOR,
    entrada_a_features,
    nivel_riesgo,
    predecir,
)

# ── Fixture: mini pipeline con las 25 features reales ────────────────────────

@pytest.fixture(scope="module")
def pipeline_mini():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, len(FEATURE_NAMES))).astype(np.float64)
    y = np.array([0] * 225 + [1] * 75)
    X[:75, :3] += 2.0
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("modelo", RandomForestClassifier(n_estimators=30, random_state=0)),
    ])
    pipe.fit(X_df, y)
    return pipe


@pytest.fixture(scope="module")
def meta_mini():
    return {
        "nombre":   "RandomForest",
        "version":  "v1",
        "auc_roc":  0.85,
        "recall":   0.70,
        "f1_score": 0.55,
        "umbral":   0.30,
        "timestamp": "2026-01-01T00:00:00",
    }


def _datos_validos() -> dict:
    return {
        "semestre":                   3,
        "promedio_semestre":          3.1,
        "promedio_acumulado":         3.3,
        "tendencia_calificaciones":  -0.2,
        "creditos_matriculados":      18,
        "creditos_aprobados":         14,
        "porcentaje_avance_real":     28.4,
        "num_asignaturas_reprobadas": 2,
        "tasa_reprobacion":           0.15,
        "semestres_cursados":         3,
        "interrupciones_previas":     0,
        "ratio_aprobacion":           0.78,
        "deficit_creditos":           5.0,
        "score_riesgo_bruto":         0.42,
        "anio_cohorte":               2022,
        "semestre_cohorte":           1,
    }


# ── Tests de sintaxis ─────────────────────────────────────────────────────────

DASHBOARD_FILES = [
    "src/sapde/dashboard/utils.py",
    "src/sapde/dashboard/app.py",
    "src/sapde/dashboard/pages/1_Prediccion.py",
    "src/sapde/dashboard/pages/2_Estudiantes_en_Riesgo.py",
    "src/sapde/dashboard/pages/3_Metricas_Modelos.py",
]

@pytest.mark.parametrize("ruta", DASHBOARD_FILES)
def test_sintaxis_archivo(ruta):
    """Todos los archivos del dashboard deben tener sintaxis Python válida."""
    raiz = Path(__file__).resolve().parents[1]
    contenido = (raiz / ruta).read_text(encoding="utf-8")
    ast.parse(contenido)


# ── Tests de constantes ───────────────────────────────────────────────────────

class TestConstantes:
    def test_feature_names_longitud(self):
        assert len(FEATURE_NAMES) == 25

    def test_feature_names_no_duplicados(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_features_incluye_semestre(self):
        assert "semestre" in FEATURE_NAMES

    def test_features_incluye_avanzadas(self):
        avanzadas = ["nota_x_avance", "repro_x_deficit", "semestre_sq", "n_factores_riesgo"]
        for f in avanzadas:
            assert f in FEATURE_NAMES, f"Feature avanzada ausente: {f}"

    def test_niveles_color_completo(self):
        for nivel in ["bajo", "moderado", "alto", "muy_alto"]:
            assert nivel in NIVELES_COLOR
            assert nivel in NIVEL_EMOJI

    def test_colores_son_hex(self):
        for color in NIVELES_COLOR.values():
            assert color.startswith("#"), f"Color no es hex: {color}"


# ── Tests de nivel_riesgo ─────────────────────────────────────────────────────

class TestNivelRiesgo:
    def test_bajo(self):
        assert nivel_riesgo(0.05, 0.30) == "bajo"
        assert nivel_riesgo(0.19, 0.30) == "bajo"

    def test_moderado(self):
        assert nivel_riesgo(0.20, 0.30) == "moderado"
        assert nivel_riesgo(0.29, 0.30) == "moderado"

    def test_alto(self):
        assert nivel_riesgo(0.30, 0.30) == "alto"
        assert nivel_riesgo(0.60, 0.30) == "alto"

    def test_muy_alto(self):
        assert nivel_riesgo(0.70, 0.30) == "muy_alto"
        assert nivel_riesgo(0.99, 0.30) == "muy_alto"

    def test_umbral_dinamico(self):
        # Con umbral 0.5, P=0.40 es moderado
        assert nivel_riesgo(0.40, 0.50) == "moderado"
        # Con umbral 0.5, P=0.50 es alto
        assert nivel_riesgo(0.50, 0.50) == "alto"


# ── Tests de entrada_a_features ───────────────────────────────────────────────

class TestEntradaAFeatures:
    def test_retorna_array_2d(self):
        X = entrada_a_features(_datos_validos(), FEATURE_NAMES)
        assert X.ndim == 2
        assert X.shape == (1, len(FEATURE_NAMES))

    def test_dtype_float(self):
        X = entrada_a_features(_datos_validos(), FEATURE_NAMES)
        assert X.dtype == np.float64

    def test_columnas_en_orden(self):
        datos = _datos_validos()
        X = entrada_a_features(datos, FEATURE_NAMES)
        # El primer feature es 'semestre' → debe coincidir
        idx_sem = FEATURE_NAMES.index("semestre")
        assert X[0, idx_sem] == pytest.approx(datos["semestre"])

    def test_features_avanzadas_computadas(self):
        datos = _datos_validos()
        X = entrada_a_features(datos, FEATURE_NAMES)
        # nota_x_avance = promedio_acumulado * porcentaje_avance_real / 100
        idx = FEATURE_NAMES.index("nota_x_avance")
        esperado = datos["promedio_acumulado"] * datos["porcentaje_avance_real"] / 100.0
        assert X[0, idx] == pytest.approx(esperado, rel=1e-5)

    def test_semestre_sq_computado(self):
        datos = _datos_validos()
        X = entrada_a_features(datos, FEATURE_NAMES)
        idx = FEATURE_NAMES.index("semestre_sq")
        assert X[0, idx] == pytest.approx(datos["semestre"] ** 2)

    def test_sin_nan(self):
        X = entrada_a_features(_datos_validos(), FEATURE_NAMES)
        assert not np.isnan(X).any()


# ── Tests de predecir ─────────────────────────────────────────────────────────

class TestPredecir:
    def test_retorna_dict_con_claves(self, pipeline_mini, meta_mini):
        resultado = predecir(_datos_validos(), pipeline_mini, meta_mini, FEATURE_NAMES)
        for clave in ["probabilidad", "prediccion", "nivel_riesgo", "umbral", "modelo", "X"]:
            assert clave in resultado

    def test_probabilidad_en_rango(self, pipeline_mini, meta_mini):
        resultado = predecir(_datos_validos(), pipeline_mini, meta_mini, FEATURE_NAMES)
        assert 0.0 <= resultado["probabilidad"] <= 1.0

    def test_nivel_riesgo_valido(self, pipeline_mini, meta_mini):
        resultado = predecir(_datos_validos(), pipeline_mini, meta_mini, FEATURE_NAMES)
        assert resultado["nivel_riesgo"] in {"bajo", "moderado", "alto", "muy_alto"}

    def test_prediccion_coherente_con_umbral(self, pipeline_mini, meta_mini):
        resultado = predecir(_datos_validos(), pipeline_mini, meta_mini, FEATURE_NAMES)
        proba = resultado["probabilidad"]
        umbral = resultado["umbral"]
        assert resultado["prediccion"] == (proba >= umbral)

    def test_umbral_proviene_de_meta(self, pipeline_mini, meta_mini):
        resultado = predecir(_datos_validos(), pipeline_mini, meta_mini, FEATURE_NAMES)
        assert resultado["umbral"] == pytest.approx(meta_mini["umbral"])

    def test_modelo_nombre_correcto(self, pipeline_mini, meta_mini):
        resultado = predecir(_datos_validos(), pipeline_mini, meta_mini, FEATURE_NAMES)
        assert resultado["modelo"] == meta_mini["nombre"]

    def test_X_shape_correcto(self, pipeline_mini, meta_mini):
        resultado = predecir(_datos_validos(), pipeline_mini, meta_mini, FEATURE_NAMES)
        assert resultado["X"].shape == (len(FEATURE_NAMES),)


# ── Tests de artefactos disponibles ──────────────────────────────────────────

class TestArtefactosDisponibles:
    def test_hay_modelos_entrenados(self):
        from sapde.dashboard.utils import cargar_todos_artefactos
        artefactos = cargar_todos_artefactos()
        assert len(artefactos) > 0, "No hay modelos entrenados en models/"

    def test_artefactos_tienen_metricas(self):
        from sapde.dashboard.utils import cargar_todos_artefactos
        artefactos = cargar_todos_artefactos()
        for art in artefactos:
            assert "auc_roc" in art
            assert "recall" in art
            assert art["auc_roc"] is not None

    def test_mejor_auc_supera_aleatorio(self):
        from sapde.dashboard.utils import cargar_todos_artefactos
        artefactos = cargar_todos_artefactos()
        mejor_auc = max(a["auc_roc"] for a in artefactos)
        assert mejor_auc > 0.7, f"AUC demasiado bajo: {mejor_auc}"

    def test_shap_resumen_tiene_modelos(self):
        from sapde.dashboard.utils import cargar_shap_resumen
        resumen = cargar_shap_resumen()
        assert len(resumen) > 0, "No hay reporte SHAP generado"
        for modelo, top_list in resumen.items():
            assert len(top_list) > 0
            assert "feature" in top_list[0]
            assert "importancia_media" in top_list[0]
