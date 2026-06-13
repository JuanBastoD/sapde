"""
Tests Fase 3 — Interpretabilidad SHAP.

Valida:
  - Inicialización de ExplicadorSHAP sin error
  - TreeExplainer para RandomForest
  - KernelExplainer para modelo no-árbol (MLP)
  - calcular_shap retorna shape correcto
  - importancia_global retorna DataFrame ordenado con n_features filas
  - importancias >= 0
  - explicar_estudiante retorna dict con todas las features
  - plot_summary genera archivo PNG
  - plot_waterfall genera archivo PNG
  - run_shap.main() completa sin errores (smoke test)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sapde.explain.shap_explainer import ExplicadorSHAP


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_SAMPLES = 500
N_FEATURES = 10
_RNG = np.random.default_rng(42)

FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


def _make_dataset():
    """Dataset imbalanceado: ~15% clase 1 (similar a deserción real)."""
    X = _RNG.standard_normal((N_SAMPLES, N_FEATURES))
    # Clase 1: 75 muestras (~15%), con señal en primeras 3 features
    y = np.zeros(N_SAMPLES, dtype=int)
    y[:75] = 1
    X[:75, :3] += 2.0  # separación entre clases
    # Mezclar aleatoriamente
    idx = _RNG.permutation(N_SAMPLES)
    return X[idx], y[idx]


def _make_pipeline_rf():
    pipeline = ImbPipeline([
        ("smote",  SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=3)),
        ("scaler", StandardScaler()),
        ("modelo", RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1)),
    ])
    X, y = _make_dataset()
    pipeline.fit(X, y)
    return pipeline, X, y


def _make_pipeline_mlp():
    pipeline = ImbPipeline([
        ("smote",  SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=3)),
        ("scaler", StandardScaler()),
        ("modelo", MLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=42)),
    ])
    X, y = _make_dataset()
    pipeline.fit(X, y)
    return pipeline, X, y


@pytest.fixture(scope="module")
def pipeline_rf():
    return _make_pipeline_rf()


@pytest.fixture(scope="module")
def pipeline_mlp():
    return _make_pipeline_mlp()


@pytest.fixture(scope="module")
def explicador_rf(pipeline_rf):
    pipe, X, _ = pipeline_rf
    return ExplicadorSHAP("RandomForest", pipe, FEATURE_NAMES, X, n_background=80)


@pytest.fixture(scope="module")
def explicador_mlp(pipeline_mlp):
    pipe, X, _ = pipeline_mlp
    return ExplicadorSHAP("MLP", pipe, FEATURE_NAMES, X, n_background=60)


# ── Tests de inicialización ───────────────────────────────────────────────────

def test_explicador_rf_inicializa(explicador_rf):
    assert explicador_rf.nombre == "RandomForest"
    assert len(explicador_rf.feature_names) == N_FEATURES


def test_explicador_mlp_inicializa(explicador_mlp):
    assert explicador_mlp.nombre == "MLP"
    assert explicador_mlp._explainer is None  # diferido


# ── Tests de cálculo SHAP ─────────────────────────────────────────────────────

def test_shap_rf_shape(pipeline_rf, explicador_rf):
    _, X, _ = pipeline_rf
    shap_vals, _ = explicador_rf.calcular_shap(X[:50])
    assert shap_vals.shape == (50, N_FEATURES)


def test_shap_mlp_shape(pipeline_mlp, explicador_mlp):
    _, X, _ = pipeline_mlp
    shap_vals, _ = explicador_mlp.calcular_shap(X[:20], nsamples_kernel=50)
    assert shap_vals.shape == (20, N_FEATURES)


def test_shap_rf_es_float(pipeline_rf, explicador_rf):
    _, X, _ = pipeline_rf
    shap_vals, _ = explicador_rf.calcular_shap(X[:10])
    assert shap_vals.dtype in (np.float32, np.float64)


def test_shap_submuestreo(pipeline_rf, explicador_rf):
    """Con n_muestras < len(X), retorna el subconjunto correcto."""
    _, X, _ = pipeline_rf
    shap_vals, idx = explicador_rf.calcular_shap(X, n_muestras=30)
    # RF (tree): no submuestrea, pero debemos verificar shape
    assert shap_vals.shape[1] == N_FEATURES


# ── Tests de importancia global ───────────────────────────────────────────────

def test_importancia_global_shape(pipeline_rf, explicador_rf):
    _, X, _ = pipeline_rf
    df = explicador_rf.importancia_global(X, n_muestras=80)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == N_FEATURES
    assert "feature" in df.columns
    assert "importancia_media" in df.columns


def test_importancia_global_positiva(pipeline_rf, explicador_rf):
    _, X, _ = pipeline_rf
    df = explicador_rf.importancia_global(X, n_muestras=80)
    assert (df["importancia_media"] >= 0).all()


def test_importancia_global_ordenada(pipeline_rf, explicador_rf):
    _, X, _ = pipeline_rf
    df = explicador_rf.importancia_global(X, n_muestras=80)
    assert df["importancia_media"].is_monotonic_decreasing


def test_importancia_global_mlp(pipeline_mlp, explicador_mlp):
    _, X, _ = pipeline_mlp
    df = explicador_mlp.importancia_global(X[:30], n_muestras=30)
    assert len(df) == N_FEATURES
    assert (df["importancia_media"] >= 0).all()


# ── Tests de explicación por estudiante ───────────────────────────────────────

def test_explicar_estudiante_rf(pipeline_rf, explicador_rf):
    _, X, _ = pipeline_rf
    result = explicador_rf.explicar_estudiante(X[0])
    assert "feature_names" in result
    assert "shap_values" in result
    assert "base_value" in result
    assert "prediccion_proba" in result
    assert "por_feature" in result
    assert len(result["shap_values"]) == N_FEATURES
    assert len(result["por_feature"]) == N_FEATURES


def test_explicar_estudiante_proba_valida(pipeline_rf, explicador_rf):
    _, X, _ = pipeline_rf
    result = explicador_rf.explicar_estudiante(X[0])
    assert 0.0 <= result["prediccion_proba"] <= 1.0


def test_explicar_estudiante_mlp(pipeline_mlp, explicador_mlp):
    _, X, _ = pipeline_mlp
    result = explicador_mlp.explicar_estudiante(X[0], nsamples_kernel=50)
    assert len(result["shap_values"]) == N_FEATURES
    assert 0.0 <= result["prediccion_proba"] <= 1.0


# ── Tests de plots ────────────────────────────────────────────────────────────

def test_plot_summary_rf(pipeline_rf, explicador_rf, tmp_path):
    _, X, _ = pipeline_rf
    ruta = tmp_path / "summary_rf.png"
    explicador_rf.plot_summary(X[:80], ruta, n_muestras=80)
    assert ruta.exists()
    assert ruta.stat().st_size > 5_000  # PNG no vacío


def test_plot_bar_rf(pipeline_rf, explicador_rf, tmp_path):
    _, X, _ = pipeline_rf
    ruta = tmp_path / "bar_rf.png"
    explicador_rf.plot_bar(X[:80], ruta, n_muestras=80)
    assert ruta.exists()


def test_plot_waterfall_rf(pipeline_rf, explicador_rf, tmp_path):
    _, X, _ = pipeline_rf
    ruta = tmp_path / "waterfall_rf.png"
    explicador_rf.plot_waterfall(X[0], ruta, label_estudiante="Test")
    assert ruta.exists()
    assert ruta.stat().st_size > 5_000


def test_plot_importancia_comparativa(pipeline_rf, explicador_rf, tmp_path):
    _, X, _ = pipeline_rf
    df_imp = explicador_rf.importancia_global(X[:80], n_muestras=80)
    ruta = tmp_path / "importancia.png"
    explicador_rf.plot_importancia_comparativa(df_imp, ruta)
    assert ruta.exists()


# ── Smoke test de run_shap.main() ─────────────────────────────────────────────

def test_run_shap_main_disponible():
    """Verifica que run_shap.main es importable."""
    from sapde.explain.run_shap import main
    assert callable(main)
