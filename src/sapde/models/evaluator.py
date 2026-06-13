"""
Evaluación de modelos para SAPDE (OE2 / OE5).

Métricas: Precisión, Recall, F1-score, AUC-ROC, AUC-PR.
Calibración de umbral: Youden's J (maximiza recall - FPR).
Salidas: tabla comparativa + gráficas (curvas ROC, PR, matrices de confusión).
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("sapde.models.evaluator")

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


# ── Calibración de umbral de decisión ────────────────────────────────────────

def calibrar_umbral_youden(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Umbral óptimo via índice J de Youden: max(TPR - FPR).

    Prioriza el recall (minimize false negatives = alumnos en riesgo no detectados).
    """
    fpr, tpr, umbrales = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    idx_opt  = np.argmax(j_scores)
    umbral   = float(umbrales[idx_opt])
    logger.info(
        "Umbral Youden: %.3f  (TPR=%.3f, FPR=%.3f)",
        umbral, tpr[idx_opt], fpr[idx_opt],
    )
    return umbral


def calibrar_umbral_recall_objetivo(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recall_objetivo: float = 0.80,
) -> float:
    """
    Encuentra el umbral más alto que alcanza un recall mínimo.

    Útil cuando se prioriza identificar a todos los desertores (minimize FN).
    """
    prec, rec, umbrales = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve incluye un umbral extra al final; ajustamos
    validos = umbrales[rec[:-1] >= recall_objetivo]
    umbral  = float(validos.max()) if len(validos) > 0 else 0.5
    idx     = np.searchsorted(umbrales, umbral)
    logger.info(
        "Umbral recall≥%.0f%%: %.3f  (prec=%.3f, rec=%.3f)",
        recall_objetivo * 100, umbral,
        prec[idx] if idx < len(prec) else 0,
        rec[idx]  if idx < len(rec)  else 0,
    )
    return umbral


# ── Evaluación completa de un modelo ─────────────────────────────────────────

def evaluar_modelo(
    nombre: str,
    pipeline: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    umbral: float = 0.5,
    t_entrenamiento: float = 0.0,
) -> dict:
    """
    Calcula el conjunto completo de métricas OE2 para un pipeline entrenado.

    Retorna dict con las métricas para armar la tabla comparativa.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= umbral).astype(int)

    precision  = float(precision_score(y_test, y_pred, zero_division=0))
    recall     = float(recall_score(y_test, y_pred, zero_division=0))
    f1         = float(f1_score(y_test, y_pred, zero_division=0))
    accuracy   = float(accuracy_score(y_test, y_pred))
    auc_roc    = float(roc_auc_score(y_test, y_proba))
    auc_pr     = float(average_precision_score(y_test, y_proba))
    cm         = confusion_matrix(y_test, y_pred)

    metricas = {
        "modelo":            nombre,
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1, 4),
        "accuracy":          round(accuracy, 4),
        "auc_roc":           round(auc_roc, 4),
        "auc_pr":            round(auc_pr, 4),
        "umbral":            round(umbral, 4),
        "t_entrenamiento_s": round(t_entrenamiento, 3),
        "TP": int(cm[1, 1]),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
    }

    logger.info(
        "%s | P=%.3f R=%.3f F1=%.3f AUC=%.3f (umbral=%.3f)",
        nombre, precision, recall, f1, auc_roc, umbral,
    )
    return metricas


# ── Tabla comparativa ─────────────────────────────────────────────────────────

def tabla_comparacion(resultados: list[dict]) -> pd.DataFrame:
    """
    Construye y formatea la tabla comparativa de todos los modelos.
    Ordenada de mayor a menor AUC-ROC.
    """
    df = pd.DataFrame(resultados)
    df = df.sort_values("auc_roc", ascending=False).reset_index(drop=True)
    df.index += 1  # ranking empieza en 1
    return df


# ── Gráficas de evaluación ────────────────────────────────────────────────────

def plot_curvas_roc(
    resultados_curvas: dict[str, tuple],
    ruta: Path,
) -> None:
    """
    Curvas ROC de todos los modelos en un solo gráfico.

    resultados_curvas: {nombre: (y_test, y_proba)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatorio (AUC=0.50)")

    colores = plt.cm.tab10.colors
    for (nombre, (y_test, y_proba)), color in zip(resultados_curvas.items(), colores):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{nombre} (AUC={auc_val:.3f})")

    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curvas ROC — Comparación de modelos SAPDE", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


def plot_curvas_pr(
    resultados_curvas: dict[str, tuple],
    ruta: Path,
) -> None:
    """Curvas Precision-Recall de todos los modelos."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colores = plt.cm.tab10.colors
    for (nombre, (y_test, y_proba)), color in zip(resultados_curvas.items(), colores):
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{nombre} (AUC-PR={auc_pr:.3f})")

    # Línea base (tasa de positivos en el dataset)
    tasa_pos = y_test.mean()
    ax.axhline(tasa_pos, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (={tasa_pos:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — Comparacion de modelos SAPDE",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


def plot_matrices_confusion(
    resultados_metricas: list[dict],
    ruta: Path,
) -> None:
    """Grid de matrices de confusión (una por modelo)."""
    n = len(resultados_metricas)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, met in enumerate(resultados_metricas):
        ax  = axes[idx // cols][idx % cols]
        cm  = np.array([[met["TN"], met["FP"]], [met["FN"], met["TP"]]])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Permanece", "Deserta"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{met['modelo']}\nP={met['precision']:.2f} R={met['recall']:.2f} "
            f"F1={met['f1_score']:.2f}",
            fontsize=9,
        )

    # Ocultar ejes sobrantes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Matrices de Confusion — Comparacion de modelos", fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


def plot_comparacion_barras(tabla: pd.DataFrame, ruta: Path) -> None:
    """Barras comparativas de métricas clave por modelo."""
    metricas_plot = ["precision", "recall", "f1_score", "auc_roc"]
    n_metricas = len(metricas_plot)
    n_modelos  = len(tabla)

    x = np.arange(n_metricas)
    ancho = 0.8 / n_modelos
    colores = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (_, fila) in enumerate(tabla.iterrows()):
        vals = [fila[m] for m in metricas_plot]
        offset = (i - n_modelos / 2 + 0.5) * ancho
        bars = ax.bar(x + offset, vals, ancho * 0.9,
                      label=fila["modelo"], color=colores[i % 10], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-Score", "AUC-ROC"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparacion de modelos — métricas OE2", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


def plot_benchmark_entrenamiento(benchmark: list[dict], ruta: Path) -> None:
    """Gráfica de speedup vs n_jobs para los modelos paralelizados (OE5)."""
    if not benchmark:
        return

    df = pd.DataFrame(benchmark)
    modelos = df["modelo"].unique()
    colores = plt.cm.tab10.colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Benchmark de paralelizacion en entrenamiento (OE3/OE5)",
                 fontweight="bold")

    for i, modelo in enumerate(modelos):
        sub = df[df["modelo"] == modelo].sort_values("n_jobs")
        ax1.plot(sub["n_jobs"].abs(), sub["t_entrenamiento_s"],
                 marker="o", label=modelo, color=colores[i])
        ax2.plot(sub["n_jobs"].abs(), sub["speedup"],
                 marker="s", label=modelo, color=colores[i])

    # Línea de speedup ideal
    n_max = df["n_jobs"].abs().max()
    ax2.plot([1, n_max], [1, n_max], "k--", linewidth=1, label="Speedup ideal")

    ax1.set_xlabel("Numero de trabajadores (n_jobs)")
    ax1.set_ylabel("Tiempo de entrenamiento (s)")
    ax1.set_title("Tiempo de entrenamiento")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Numero de trabajadores (n_jobs)")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup vs n_jobs")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)
