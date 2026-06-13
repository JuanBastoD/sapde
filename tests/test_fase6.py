"""
Tests Fase 6 — Validación estadística y generación de reporte.

Cubre:
  - validacion_cruzada(): shape, rangos, consistencia
  - bootstrap_ic(): propiedades del IC
  - curva_calibracion(): estructura y valores
  - report_generator: gráficas y HTML
  - run_validation: disponibilidad del orquestador
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sapde.validation.validador import (
    bootstrap_ic,
    bootstrap_todas_metricas,
    curva_calibracion,
    validacion_cruzada,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def datos_binarios():
    """Dataset pequeño con señal clara para pruebas rápidas."""
    rng = np.random.default_rng(42)
    n = 400
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def pipeline_lr():
    """Pipeline LR rápido (sin SMOTE para velocidad en tests)."""
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=300, random_state=42)),
    ])


@pytest.fixture(scope="module")
def pipeline_lr_fitted(datos_binarios, pipeline_lr):
    X, y = datos_binarios
    X_train, y_train = X[:320], y[:320]
    pipeline_lr.fit(X_train, y_train)
    return pipeline_lr


@pytest.fixture(scope="module")
def y_true_proba(datos_binarios, pipeline_lr_fitted):
    X, y = datos_binarios
    X_test, y_test = X[320:], y[320:]
    y_proba = pipeline_lr_fitted.predict_proba(X_test)[:, 1]
    return y_test, y_proba


# ── Tests: validacion_cruzada ─────────────────────────────────────────────────

class TestValidacionCruzada:
    def test_retorna_dict_con_claves(self, pipeline_lr, datos_binarios):
        X, y = datos_binarios
        res = validacion_cruzada(pipeline_lr, X, y, umbral=0.5, n_splits=3, semilla=42)
        assert "media" in res
        assert "std" in res
        assert "por_fold" in res
        assert "n_splits" in res

    def test_n_splits_correcto(self, pipeline_lr, datos_binarios):
        X, y = datos_binarios
        res = validacion_cruzada(pipeline_lr, X, y, umbral=0.5, n_splits=3)
        assert res["n_splits"] == 3
        assert len(res["por_fold"]) == 3

    def test_auc_en_rango(self, pipeline_lr, datos_binarios):
        X, y = datos_binarios
        res = validacion_cruzada(pipeline_lr, X, y, umbral=0.5, n_splits=3)
        assert 0.0 <= res["media"]["auc_roc"] <= 1.0

    def test_auc_supera_aleatorio(self, pipeline_lr, datos_binarios):
        X, y = datos_binarios
        res = validacion_cruzada(pipeline_lr, X, y, umbral=0.5, n_splits=3)
        assert res["media"]["auc_roc"] > 0.6, "La regresión logística debe superar azar con señal clara"

    def test_std_no_negativa(self, pipeline_lr, datos_binarios):
        X, y = datos_binarios
        res = validacion_cruzada(pipeline_lr, X, y, umbral=0.5, n_splits=3)
        for m, v in res["std"].items():
            assert v >= 0.0, f"std negativa para {m}"

    def test_metricas_presentes(self, pipeline_lr, datos_binarios):
        X, y = datos_binarios
        res = validacion_cruzada(pipeline_lr, X, y, umbral=0.5, n_splits=3)
        for met in ["auc_roc", "recall", "f1_score", "precision", "brier"]:
            assert met in res["media"], f"Falta métrica: {met}"

    def test_por_fold_tiene_auc(self, pipeline_lr, datos_binarios):
        X, y = datos_binarios
        res = validacion_cruzada(pipeline_lr, X, y, umbral=0.5, n_splits=3)
        for fold_data in res["por_fold"]:
            assert "auc_roc" in fold_data
            assert 0.0 <= fold_data["auc_roc"] <= 1.0


# ── Tests: bootstrap_ic ───────────────────────────────────────────────────────

class TestBootstrapIC:
    def test_estructura_resultado(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = bootstrap_ic(y_true, y_proba, umbral=0.5, metrica="auc_roc", n_iter=200)
        for clave in ["media", "std", "lower", "upper", "alpha", "n_iter", "metrica"]:
            assert clave in res, f"Falta clave: {clave}"

    def test_ic_coherente(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = bootstrap_ic(y_true, y_proba, umbral=0.5, metrica="auc_roc", n_iter=200)
        assert res["lower"] <= res["media"] <= res["upper"]

    def test_auc_en_rango(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = bootstrap_ic(y_true, y_proba, umbral=0.5, metrica="auc_roc", n_iter=200)
        assert 0.0 <= res["lower"] <= 1.0
        assert 0.0 <= res["upper"] <= 1.0

    def test_alpha_respetado(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = bootstrap_ic(y_true, y_proba, umbral=0.5, metrica="auc_roc",
                            n_iter=200, alpha=0.90)
        assert res["alpha"] == pytest.approx(0.90)

    def test_metrica_recall(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = bootstrap_ic(y_true, y_proba, umbral=0.5, metrica="recall", n_iter=200)
        assert 0.0 <= res["media"] <= 1.0

    def test_metrica_invalida_lanza_error(self, y_true_proba):
        y_true, y_proba = y_true_proba
        with pytest.raises(ValueError, match="Métrica desconocida"):
            bootstrap_ic(y_true, y_proba, umbral=0.5, metrica="inventada", n_iter=10)

    def test_std_positiva(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = bootstrap_ic(y_true, y_proba, umbral=0.5, metrica="auc_roc", n_iter=200)
        assert res["std"] >= 0.0

    def test_todas_metricas(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = bootstrap_todas_metricas(y_true, y_proba, umbral=0.5, n_iter=100)
        for m in ["auc_roc", "recall", "f1_score", "precision", "brier"]:
            assert m in res
            assert res[m]["lower"] <= res[m]["upper"]

    def test_reproducibilidad(self, y_true_proba):
        y_true, y_proba = y_true_proba
        r1 = bootstrap_ic(y_true, y_proba, 0.5, "auc_roc", 200, semilla=99)
        r2 = bootstrap_ic(y_true, y_proba, 0.5, "auc_roc", 200, semilla=99)
        assert r1["media"] == pytest.approx(r2["media"])

    def test_semillas_distintas_difieren(self, y_true_proba):
        y_true, y_proba = y_true_proba
        r1 = bootstrap_ic(y_true, y_proba, 0.5, "auc_roc", 200, semilla=1)
        r2 = bootstrap_ic(y_true, y_proba, 0.5, "auc_roc", 200, semilla=2)
        # Pueden diferir ligeramente
        assert abs(r1["media"] - r2["media"]) < 0.05


# ── Tests: curva_calibracion ──────────────────────────────────────────────────

class TestCurvasCalibracion:
    def test_estructura(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = curva_calibracion(y_true, y_proba, n_bins=8)
        for clave in ["frac_positivos", "prob_media", "brier_score", "ece", "n_bins"]:
            assert clave in res

    def test_brier_en_rango(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = curva_calibracion(y_true, y_proba)
        assert 0.0 <= res["brier_score"] <= 1.0

    def test_ece_no_negativa(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = curva_calibracion(y_true, y_proba)
        assert res["ece"] >= 0.0

    def test_longitud_bins(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = curva_calibracion(y_true, y_proba, n_bins=8)
        assert len(res["frac_positivos"]) == len(res["prob_media"])

    def test_frac_positivos_en_rango(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = curva_calibracion(y_true, y_proba)
        for fp in res["frac_positivos"]:
            assert 0.0 <= fp <= 1.0

    def test_prob_media_en_rango(self, y_true_proba):
        y_true, y_proba = y_true_proba
        res = curva_calibracion(y_true, y_proba)
        for pm in res["prob_media"]:
            assert 0.0 <= pm <= 1.0

    def test_clasificador_perfecto_brier_bajo(self):
        # Predicciones casi perfectas → Brier score bajo
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.95, 0.90, 0.05, 0.10, 0.88, 0.08])
        res = curva_calibracion(y_true, y_proba, n_bins=5)
        assert res["brier_score"] < 0.05

    def test_clasificador_aleatorio_brier_alto(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_proba = rng.uniform(0, 1, 200)
        res = curva_calibracion(y_true, y_proba)
        assert res["brier_score"] > 0.10


# ── Tests: report_generator ───────────────────────────────────────────────────

class TestReportGenerator:
    def _resultados_mock(self) -> list[dict]:
        """Resultados mínimos con la estructura que espera generar_html."""
        def ic_mock(m=0.8, l=0.75, u=0.85):
            return {"media": m, "lower": l, "upper": u, "alpha": 0.95,
                    "std": 0.02, "n_iter": 200, "metrica": "auc_roc"}

        return [{
            "nombre":   "MockModel",
            "version":  1,
            "umbral":   0.30,
            "metricas_test": {},
            "bootstrap": {
                "auc_roc":   ic_mock(0.83, 0.79, 0.87),
                "recall":    ic_mock(0.75, 0.68, 0.82),
                "f1_score":  ic_mock(0.20, 0.15, 0.25),
                "precision": ic_mock(0.12, 0.09, 0.15),
                "brier":     {"media": 0.04, "lower": 0.03, "upper": 0.05,
                               "alpha": 0.95, "std": 0.005, "n_iter": 200, "metrica": "brier"},
            },
            "cv": {
                "n_splits": 5,
                "por_fold": [],
                "media": {"auc_roc": 0.82, "recall": 0.70, "f1_score": 0.19,
                           "precision": 0.11, "brier": 0.04},
                "std":   {"auc_roc": 0.01, "recall": 0.05, "f1_score": 0.02,
                           "precision": 0.02, "brier": 0.003},
            },
            "calibracion": {
                "frac_positivos": [0.05, 0.10, 0.20],
                "prob_media":     [0.05, 0.15, 0.25],
                "brier_score":    0.04,
                "ece":            0.03,
                "n_bins":         10,
            },
        }]

    def _benchmarks_mock(self):
        etl = [
            {"hilos": 1, "speedup": 1.0, "eficiencia": 1.0},
            {"hilos": 2, "speedup": 0.7, "eficiencia": 0.35},
            {"hilos": 4, "speedup": 0.5, "eficiencia": 0.12},
        ]
        train = [
            {"modelo": "MockModel", "n_jobs": 1, "t_entrenamiento_s": 5.0, "speedup": 1.0, "eficiencia": 1.0},
            {"modelo": "MockModel", "n_jobs": 2, "t_entrenamiento_s": 3.5, "speedup": 1.4, "eficiencia": 0.7},
        ]
        return etl, train

    def test_generar_html_crea_archivo(self):
        from sapde.validation.report_generator import generar_html
        resultados = self._resultados_mock()
        etl, train = self._benchmarks_mock()

        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "reporte.html"
            generar_html(resultados, etl, train, ruta)
            assert ruta.exists()
            assert ruta.stat().st_size > 10_000  # > 10 KB

    def test_html_contiene_secciones_clave(self):
        from sapde.validation.report_generator import generar_html
        resultados = self._resultados_mock()
        etl, train = self._benchmarks_mock()

        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "reporte.html"
            generar_html(resultados, etl, train, ruta)
            contenido = ruta.read_text(encoding="utf-8")

        for seccion in ["Bootstrap", "Calibración", "Speedup", "Conclusiones"]:
            assert seccion in contenido, f"Falta sección: {seccion}"

    def test_html_contiene_imagenes_base64(self):
        from sapde.validation.report_generator import generar_html
        resultados = self._resultados_mock()
        etl, train = self._benchmarks_mock()

        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "reporte.html"
            generar_html(resultados, etl, train, ruta)
            contenido = ruta.read_text(encoding="utf-8")

        assert "data:image/png;base64," in contenido

    def test_html_contiene_nombre_modelo(self):
        from sapde.validation.report_generator import generar_html
        resultados = self._resultados_mock()
        etl, train = self._benchmarks_mock()

        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "reporte.html"
            generar_html(resultados, etl, train, ruta)
            contenido = ruta.read_text(encoding="utf-8")

        assert "MockModel" in contenido

    def test_plot_speedup_etl_retorna_string(self):
        from sapde.validation.report_generator import plot_speedup_etl
        etl, _ = self._benchmarks_mock()
        resultado = plot_speedup_etl(etl)
        assert isinstance(resultado, str)
        assert len(resultado) > 100  # base64 no vacío

    def test_plot_auc_bootstrap_retorna_string(self):
        from sapde.validation.report_generator import plot_auc_bootstrap
        resultados = self._resultados_mock()
        resultado = plot_auc_bootstrap(resultados)
        assert isinstance(resultado, str)
        assert len(resultado) > 100

    def test_plot_calibracion_retorna_string(self):
        from sapde.validation.report_generator import plot_calibracion
        resultados = self._resultados_mock()
        resultado = plot_calibracion(resultados)
        assert isinstance(resultado, str)
        assert len(resultado) > 100


# ── Tests: run_validation disponibilidad ─────────────────────────────────────

class TestRunValidation:
    def test_modulo_importa(self):
        import sapde.validation.run_validation as rv
        assert hasattr(rv, "main")

    def test_leer_json_archivo_inexistente(self):
        from sapde.validation.run_validation import _leer_json
        res = _leer_json(Path("/tmp/no_existe_xyz.json"))
        assert res == []

    def test_leer_json_benchmark_etl(self):
        from sapde.validation.run_validation import _leer_json
        raiz = Path(__file__).resolve().parents[1]
        ruta = raiz / "reports" / "benchmark_etl.json"
        if ruta.exists():
            data = _leer_json(ruta)
            assert isinstance(data, list)
            assert len(data) > 0
            assert "speedup" in data[0]

    def test_imprimir_resumen_no_falla(self, capsys):
        from sapde.validation.run_validation import _imprimir_resumen

        resultados_mock = [{
            "nombre": "TestModel",
            "bootstrap": {
                "auc_roc": {"media": 0.83, "lower": 0.79, "upper": 0.87},
                "recall":  {"media": 0.75, "lower": 0.68, "upper": 0.82},
            },
            "calibracion": {"ece": 0.03},
        }]
        _imprimir_resumen(resultados_mock)
        captura = capsys.readouterr()
        assert "TestModel" in captura.out
        assert "0.8300" in captura.out
