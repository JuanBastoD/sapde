"""
Validación estadística de modelos SAPDE (Fase 6).

Provee:
  - Validación cruzada estratificada (5-fold)
  - Intervalos de confianza Bootstrap (1 000 iteraciones)
  - Curvas de calibración (reliability diagram + Brier score)
  - Evaluación completa combinando todas las métricas
"""

import logging
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("sapde.validation")

# ── Funciones de métricas ─────────────────────────────────────────────────────

_METRICAS: dict[str, Callable] = {
    "auc_roc":   lambda yt, yp, u: roc_auc_score(yt, yp),
    "auc_pr":    lambda yt, yp, u: average_precision_score(yt, yp),
    "recall":    lambda yt, yp, u: recall_score(yt, (yp >= u).astype(int), zero_division=0),
    "precision": lambda yt, yp, u: precision_score(yt, (yp >= u).astype(int), zero_division=0),
    "f1_score":  lambda yt, yp, u: f1_score(yt, (yp >= u).astype(int), zero_division=0),
    "brier":     lambda yt, yp, u: brier_score_loss(yt, yp),
}


# ── Validación cruzada ────────────────────────────────────────────────────────

def validacion_cruzada(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    umbral: float,
    n_splits: int = 5,
    semilla: int = 42,
) -> dict:
    """
    Validación cruzada estratificada sobre el pipeline completo (incluye SMOTE).

    Retorna dict con mean y std de cada métrica, y los resultados por fold.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    resultados_fold: list[dict] = []

    for fold, (idx_train, idx_val) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X[idx_train], X[idx_val]
        y_tr, y_val = y[idx_train], y[idx_val]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_tr, y_tr)
            y_proba = pipeline.predict_proba(X_val)[:, 1]

        fila = {"fold": fold}
        for nombre, fn in _METRICAS.items():
            try:
                fila[nombre] = float(fn(y_val, y_proba, umbral))
            except Exception:
                fila[nombre] = float("nan")
        resultados_fold.append(fila)
        logger.debug("Fold %d: AUC=%.4f  Recall=%.4f", fold, fila["auc_roc"], fila["recall"])

    df = pd.DataFrame(resultados_fold)
    metricas_cols = [c for c in df.columns if c != "fold"]

    resumen = {
        "n_splits":     n_splits,
        "por_fold":     resultados_fold,
        "media":        {m: float(df[m].mean()) for m in metricas_cols},
        "std":          {m: float(df[m].std(ddof=1)) for m in metricas_cols},
    }
    logger.info(
        "CV %d-fold: AUC %.4f±%.4f  Recall %.4f±%.4f",
        n_splits,
        resumen["media"]["auc_roc"], resumen["std"]["auc_roc"],
        resumen["media"]["recall"],  resumen["std"]["recall"],
    )
    return resumen


# ── Bootstrap con intervalo de confianza ──────────────────────────────────────

def bootstrap_ic(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    umbral: float,
    metrica: str = "auc_roc",
    n_iter: int = 1_000,
    alpha: float = 0.95,
    semilla: int = 42,
) -> dict:
    """
    Calcula el IC Bootstrap (sin reentrenamiento) para la métrica indicada.

    Retorna dict con media, lower, upper, std (en el conjunto de test dado).
    """
    if metrica not in _METRICAS:
        raise ValueError(f"Métrica desconocida: {metrica}. Opciones: {list(_METRICAS)}")

    fn = _METRICAS[metrica]
    rng = np.random.default_rng(semilla)
    n = len(y_true)
    muestras = np.empty(n_iter)

    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_proba[idx]
        # Evitar folds sin positivos (no se puede calcular AUC)
        if yt.sum() == 0 or yt.sum() == len(yt):
            muestras[i] = float("nan")
        else:
            try:
                muestras[i] = fn(yt, yp, umbral)
            except Exception:
                muestras[i] = float("nan")

    muestras = muestras[~np.isnan(muestras)]
    tail = (1 - alpha) / 2
    resultado = {
        "metrica":  metrica,
        "media":    float(np.mean(muestras)),
        "std":      float(np.std(muestras, ddof=1)),
        "lower":    float(np.quantile(muestras, tail)),
        "upper":    float(np.quantile(muestras, 1 - tail)),
        "alpha":    alpha,
        "n_iter":   len(muestras),
    }
    logger.debug(
        "Bootstrap %s: %.4f [%.4f, %.4f] (α=%.0f%%)",
        metrica, resultado["media"], resultado["lower"], resultado["upper"], alpha * 100,
    )
    return resultado


def bootstrap_todas_metricas(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    umbral: float,
    n_iter: int = 1_000,
    alpha: float = 0.95,
    semilla: int = 42,
) -> dict[str, dict]:
    """Aplica bootstrap a todas las métricas definidas en _METRICAS."""
    return {
        m: bootstrap_ic(y_true, y_proba, umbral, m, n_iter, alpha, semilla)
        for m in _METRICAS
    }


# ── Curva de calibración ──────────────────────────────────────────────────────

def curva_calibracion(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Calcula los datos de la curva de calibración (reliability diagram).

    Retorna dict con: frac_positivos, prob_media, brier_score, ece.
    ECE = Expected Calibration Error (promedio ponderado de |gap|).
    """
    frac_pos, prob_med = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, y_proba))

    # ECE ponderado por el tamaño de cada bin
    n = len(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    pesos = []
    gaps  = []
    for lo, hi, fp, pm in zip(bins[:-1], bins[1:], frac_pos, prob_med):
        mask = (y_proba >= lo) & (y_proba < hi)
        peso = mask.sum() / n
        pesos.append(peso)
        gaps.append(abs(fp - pm))
    ece = float(np.dot(pesos, gaps))

    return {
        "frac_positivos": frac_pos.tolist(),
        "prob_media":     prob_med.tolist(),
        "brier_score":    brier,
        "ece":            ece,
        "n_bins":         n_bins,
    }


# ── Evaluación integral de todos los modelos ─────────────────────────────────

def evaluar_modelos_completo(
    artefactos: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_splits: int = 5,
    n_bootstrap: int = 500,
    semilla: int = 42,
) -> list[dict]:
    """
    Ejecuta la validación completa para cada artefacto.

    Por cada modelo:
      - Carga el pipeline entrenado
      - Calcula probabilidades en test
      - Bootstrap IC en test
      - Validación cruzada en train+test (completo)
      - Curva de calibración

    Retorna lista de dicts con todos los resultados.
    """
    from sapde.models.artifacts import cargar_pipeline
    from pathlib import Path

    _MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    resultados = []
    for art in artefactos:
        nombre, version = art["nombre"], art["version"]
        logger.info("Validando %s v%s ...", nombre, version)

        try:
            pipeline, meta = cargar_pipeline(nombre, version, _MODELS_DIR)
        except Exception as exc:
            logger.warning("No se pudo cargar %s v%s: %s", nombre, version, exc)
            continue

        umbral = float(meta.get("umbral", 0.5))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_proba_test = pipeline.predict_proba(X_test)[:, 1]

        # Bootstrap IC en test set
        bootstrap = bootstrap_todas_metricas(
            y_test, y_proba_test, umbral, n_bootstrap, semilla=semilla
        )

        # Validación cruzada (pipeline completo, datos completos)
        cv = validacion_cruzada(pipeline, X_full, y_full, umbral, n_splits, semilla)

        # Calibración
        calib = curva_calibracion(y_test, y_proba_test)

        resultados.append({
            "nombre":    nombre,
            "version":   version,
            "umbral":    umbral,
            "metricas_test": art,        # métricas del CSV original
            "bootstrap": bootstrap,
            "cv":        cv,
            "calibracion": calib,
        })
        logger.info(
            "%s: AUC %.4f [%.4f, %.4f]  CV %.4f±%.4f  Brier %.4f",
            nombre,
            bootstrap["auc_roc"]["media"],
            bootstrap["auc_roc"]["lower"],
            bootstrap["auc_roc"]["upper"],
            cv["media"]["auc_roc"],
            cv["std"]["auc_roc"],
            calib["brier_score"],
        )

    return resultados
