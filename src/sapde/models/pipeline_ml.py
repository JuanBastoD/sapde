"""
Pipeline ML reproducible para SAPDE (OE2 / OE3).

Cada modelo se envuelve en un ImbPipeline de imbalanced-learn:
    SMOTE  →  StandardScaler  →  Modelo

SMOTE se aplica SOLO sobre el conjunto de entrenamiento (nunca sobre test),
garantizando que no haya fuga de datos (data leakage).

Hiperparámetros buscados con RandomizedSearchCV (n_jobs=-1 para OE3).
"""

import logging
import time
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from sapde.models.anfis import ANFISClassifier

logger = logging.getLogger("sapde.models.pipeline")

# ── Semilla global del proyecto ─────────────────────────────────────────────
_SEMILLA = 42

# ── Parámetros de cross-validation ──────────────────────────────────────────
_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=_SEMILLA)


# ── Fábrica de pipelines ─────────────────────────────────────────────────────

def crear_pipeline(
    modelo: Any,
    smote_sampling: float = 0.5,
    random_state: int = _SEMILLA,
) -> ImbPipeline:
    """
    Crea el pipeline completo: SMOTE → StandardScaler → Modelo.

    SMOTE usa sampling_strategy=0.5 (la clase minoritaria queda al 50% de la mayor).
    """
    return ImbPipeline([
        ("smote",  SMOTE(sampling_strategy=smote_sampling,
                         random_state=random_state, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("modelo", modelo),
    ])


# ── Definiciones de modelos ──────────────────────────────────────────────────

def obtener_modelos(random_state: int = _SEMILLA, n_jobs: int = -1) -> dict[str, Any]:
    """
    Retorna dict {nombre: modelo_sklearn} con todos los modelos de OE2.

    OE3: RF y LGB usan n_jobs=-1 para paralelizar el entrenamiento.
    SVM: usa CalibratedClassifierCV para obtener probabilidades.
    MLP: usa BLAS multi-hilo internamente.
    ANFIS: optimización serie (L-BFGS-B), paralelizable via CV.
    """
    try:
        import lightgbm as lgb
        lgb_disponible = True
    except ImportError:
        lgb_disponible = False
        logger.warning("LightGBM no instalado. Reemplazado por GradientBoostingClassifier.")

    modelos: dict[str, Any] = {

        # LinearSVC es O(n·p) — mucho más rápido que SVC-RBF en 10K+ muestras.
        # CalibratedClassifierCV agrega predict_proba con calibración isotónica.
        "SVM": CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            ),
            cv=3,
            ensemble=False,
        ),

        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        ),

        "ANFIS": ANFISClassifier(
            n_reglas=5,
            n_componentes_pca=6,       # reduce dimensionalidad antes del ANFIS
            max_iter=80,               # convergencia rápida para CV
            random_state=random_state,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,             # OE3: paralelización con OpenMP/BLAS
        ),
    }

    if lgb_disponible:
        import lightgbm as lgb
        modelos["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,             # OE3: paralelización
            verbosity=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        modelos["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state,
        )

    return modelos


# ── Espacios de búsqueda de hiperparámetros ──────────────────────────────────

def _espacio_svm() -> dict:
    # LinearSVC: solo C como hiperparámetro principal
    return {
        "modelo__estimator__C": [0.01, 0.1, 1.0, 10.0],
    }

def _espacio_mlp() -> dict:
    return {
        "modelo__hidden_layer_sizes": [(64,), (128, 64), (64, 32), (128, 64, 32)],
        "modelo__alpha":              [1e-4, 1e-3, 1e-2],
        "modelo__learning_rate_init": [0.001, 0.01],
    }

def _espacio_anfis() -> dict:
    return {
        "modelo__n_reglas":          [4, 5, 6],
        "modelo__n_componentes_pca": [5, 6, 8],
        "modelo__max_iter":          [60, 80],
    }

def _espacio_rf() -> dict:
    return {
        "modelo__n_estimators":     [100, 200, 300],
        "modelo__max_depth":        [None, 10, 20],
        "modelo__min_samples_leaf": [1, 2, 4],
        "modelo__max_features":     ["sqrt", "log2"],
    }

def _espacio_lgb() -> dict:
    return {
        "modelo__n_estimators":  [100, 200, 300],
        "modelo__learning_rate": [0.01, 0.05, 0.1],
        "modelo__num_leaves":    [15, 31, 63],
        "modelo__min_child_samples": [10, 20, 50],
    }

ESPACIOS_BUSQUEDA: dict[str, callable] = {
    "SVM":            _espacio_svm,
    "MLP":            _espacio_mlp,
    "ANFIS":          _espacio_anfis,
    "RandomForest":   _espacio_rf,
    "LightGBM":       _espacio_lgb,
    "GradientBoosting": _espacio_rf,
}


# ── Búsqueda de hiperparámetros ──────────────────────────────────────────────

def buscar_hiperparametros(
    nombre: str,
    pipeline: ImbPipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 15,
    n_jobs: int = -1,
    random_state: int = _SEMILLA,
) -> tuple[ImbPipeline, dict]:
    """
    Búsqueda aleatoria de hiperparámetros con cross-validation estratificado.

    OE3: n_jobs=-1 paraleliza las evaluaciones de CV en múltiples núcleos.

    Retorna
    -------
    (pipeline_optimo, mejores_params)
    """
    espacio_fn = ESPACIOS_BUSQUEDA.get(nombre)
    if espacio_fn is None:
        logger.warning("No hay espacio de búsqueda para '%s'. Usando defaults.", nombre)
        return pipeline, {}

    espacio = espacio_fn()

    logger.info("Buscando hiperparámetros para %s (%d iteraciones, CV=5)...", nombre, n_iter)
    t0 = time.perf_counter()

    busqueda = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=espacio,
        n_iter=n_iter,
        cv=_CV,
        scoring="roc_auc",       # métrica de selección: AUC-ROC
        n_jobs=n_jobs,           # OE3: paralelización de la búsqueda
        random_state=random_state,
        verbose=0,
        refit=True,              # re-entrena con los mejores parámetros
        error_score="raise",
    )

    try:
        busqueda.fit(X_train, y_train)
        t_elapsed = time.perf_counter() - t0
        logger.info(
            "%s: mejor AUC=%.4f (%.1fs) | params=%s",
            nombre, busqueda.best_score_, t_elapsed, busqueda.best_params_,
        )
        return busqueda.best_estimator_, busqueda.best_params_
    except Exception as e:
        logger.error("Error en busqueda de hiperparametros para %s: %s", nombre, e)
        logger.warning("Usando parametros por defecto para %s.", nombre)
        pipeline.fit(X_train, y_train)
        return pipeline, {}


# ── Benchmark de paralelización (OE3 / OE5) ─────────────────────────────────

def benchmark_paralelizacion(
    nombre: str,
    modelo_base: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_jobs_lista: list[int] = [1, 2, 4],
) -> list[dict]:
    """
    Mide el tiempo de entrenamiento del modelo con distintos n_jobs.
    Solo aplicable a modelos con n_jobs en su __init__ directo (RF, LGB).
    """
    import inspect
    sig = inspect.signature(modelo_base.__class__.__init__)
    if "n_jobs" not in sig.parameters:
        logger.info("%s no tiene n_jobs directo. Benchmark omitido.", nombre)
        return []

    # Solo parámetros de primer nivel (sin "__" = no son parámetros anidados)
    params_directos = {k: v for k, v in modelo_base.get_params().items()
                       if "__" not in k}

    resultados = []
    t1_ref = None

    for n in n_jobs_lista:
        modelo_copia = modelo_base.__class__(**{
            **params_directos,
            "n_jobs": n,
        })
        pipeline = crear_pipeline(modelo_copia)

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        t_total = time.perf_counter() - t0

        if t1_ref is None or n == 1:
            t1_ref = t_total

        speedup    = t1_ref / t_total if t_total > 0 else 0.0
        eficiencia = speedup / abs(n) if n != 0 else 0.0

        resultados.append({
            "modelo": nombre,
            "n_jobs": n,
            "t_entrenamiento_s": round(t_total, 4),
            "speedup": round(speedup, 3),
            "eficiencia": round(eficiencia, 3),
        })
        logger.info(
            "  %s | n_jobs=%d | t=%.3fs | speedup=%.2fx",
            nombre, n, t_total, speedup,
        )

    return resultados
