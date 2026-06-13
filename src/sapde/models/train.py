"""
Orquestador de entrenamiento SAPDE (OE2 / OE3).

Flujo:
  1. Carga datos procesados (Parquet o CSV del ETL)
  2. Feature engineering completo
  3. Split estratificado + SMOTE (dentro del pipeline)
  4. Entrena y optimiza SVM, MLP, ANFIS, RandomForest, LightGBM
  5. Calibra umbral de decisión (Youden's J)
  6. Evalúa en test set: precisión, recall, F1, AUC-ROC
  7. Genera gráficas comparativas y tabla final
  8. Guarda artefactos versionados
  9. Benchmark de paralelización (OE3/OE5)

Uso:
    python -m sapde.models.train
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sapde.config import obtener_config
from sapde.features.engineering import preparar_dataset_ml
from sapde.models.artifacts import guardar_pipeline, listar_artefactos
from sapde.models.evaluator import (
    calibrar_umbral_youden,
    evaluar_modelo,
    plot_benchmark_entrenamiento,
    plot_comparacion_barras,
    plot_curvas_pr,
    plot_curvas_roc,
    plot_matrices_confusion,
    tabla_comparacion,
)
from sapde.models.pipeline_ml import (
    benchmark_paralelizacion,
    buscar_hiperparametros,
    crear_pipeline,
    obtener_modelos,
)

logger = logging.getLogger("sapde.models.train")


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_datos_entrenamiento(cfg=None) -> pd.DataFrame:
    """
    Carga el dataset procesado (preferiblemente Parquet, si no CSV del ETL).
    Aplica feature engineering completo.
    """
    if cfg is None:
        cfg = obtener_config()
    rutas = cfg.rutas_absolutas()

    # Preferir Parquet (procesado completo), luego CSV del ETL, luego raw
    candidatos = [
        rutas["processed"] / "dataset_final.parquet",
        rutas["processed"] / "dataset_etl.parquet",
        rutas["interim"]   / "estudiantes_etl.csv",
        rutas["raw"]       / "estudiantes_sintetico.csv",
    ]

    df = None
    for ruta in candidatos:
        if ruta.exists():
            logger.info("Cargando datos desde: %s", ruta)
            df = pd.read_parquet(ruta) if ruta.suffix == ".parquet" \
                 else pd.read_csv(ruta, encoding="utf-8")
            break

    if df is None:
        raise FileNotFoundError(
            "No se encontro dataset procesado. "
            "Ejecuta: python -m sapde.etl.pipeline"
        )

    logger.info("Dataset: %d filas, %d columnas", len(df), len(df.columns))
    return df


# ── Pipeline completo de entrenamiento ───────────────────────────────────────

def entrenar_todos_los_modelos(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg=None,
    n_iter_busqueda: int = 12,
    n_jobs: int = -1,
    random_state: int = 42,
) -> tuple[list[dict], dict[str, tuple], list[dict]]:
    """
    Entrena, optimiza y evalúa todos los modelos.

    Retorna
    -------
    (metricas_lista, curvas_dict, benchmark_lista)
    """
    if cfg is None:
        cfg = obtener_config()
    rutas = cfg.rutas_absolutas()

    modelos_base = obtener_modelos(random_state=random_state, n_jobs=n_jobs)
    metricas_lista: list[dict] = []
    curvas_dict:   dict[str, tuple] = {}
    benchmark_lista: list[dict] = []

    for nombre, modelo_base in modelos_base.items():
        logger.info("=" * 50)
        logger.info("Entrenando: %s", nombre)
        logger.info("=" * 50)

        pipeline = crear_pipeline(modelo_base, random_state=random_state)

        # ── Búsqueda de hiperparámetros (OE2) ──────────────────────────────
        t0 = time.perf_counter()
        try:
            pipeline_opt, mejores_params = buscar_hiperparametros(
                nombre, pipeline, X_train, y_train,
                n_iter=n_iter_busqueda, n_jobs=n_jobs, random_state=random_state,
            )
        except Exception as e:
            logger.error("Error en busqueda para %s: %s. Usando defaults.", nombre, e)
            pipeline_opt = crear_pipeline(modelo_base, random_state=random_state)
            pipeline_opt.fit(X_train, y_train)
            mejores_params = {}
        t_entrenamiento = time.perf_counter() - t0

        # ── Calibración de umbral (Youden's J) ─────────────────────────────
        try:
            y_proba_train = pipeline_opt.predict_proba(X_train)[:, 1]
            umbral = calibrar_umbral_youden(y_train, y_proba_train)
        except Exception:
            umbral = 0.5

        # ── Evaluación en test set (OE2) ───────────────────────────────────
        try:
            metricas = evaluar_modelo(
                nombre, pipeline_opt, X_test, y_test,
                umbral=umbral, t_entrenamiento=t_entrenamiento,
            )
        except Exception as e:
            logger.error("Error evaluando %s: %s", nombre, e)
            continue

        metricas_lista.append(metricas)

        # Guardar probabilidades para curvas ROC/PR
        y_proba_test = pipeline_opt.predict_proba(X_test)[:, 1]
        curvas_dict[nombre] = (y_test, y_proba_test)

        # ── Persistencia de artefactos (OE3) ────────────────────────────────
        guardar_pipeline(
            nombre, pipeline_opt, metricas,
            umbral, mejores_params,
            directorio=rutas["models"],
        )

        # ── Benchmark de paralelización (OE3/OE5) ──────────────────────────
        if hasattr(modelo_base, "n_jobs"):
            bench = benchmark_paralelizacion(
                nombre, modelo_base, X_train, y_train,
                n_jobs_lista=[1, 2, 4],
            )
            benchmark_lista.extend(bench)

    return metricas_lista, curvas_dict, benchmark_lista


# ── Generación de reportes ────────────────────────────────────────────────────

def generar_reportes(
    metricas_lista: list[dict],
    curvas_dict: dict[str, tuple],
    benchmark_lista: list[dict],
    rutas: dict,
) -> None:
    """Genera todos los gráficos y archivos de reporte del Fase 2."""
    rep = rutas["reports"]
    rep.mkdir(parents=True, exist_ok=True)

    # Tabla comparativa
    tabla = tabla_comparacion(metricas_lista)
    logger.info("\nTabla comparativa de modelos:\n%s", tabla[
        ["modelo", "precision", "recall", "f1_score", "auc_roc", "umbral"]
    ].to_string(index=True))

    # Guardar tabla CSV
    tabla.to_csv(rep / "comparacion_modelos.csv", index=True, encoding="utf-8")

    # Gráficas
    if curvas_dict:
        y_test = list(curvas_dict.values())[0][0]
        plot_curvas_roc(curvas_dict, rep / "curvas_roc.png")
        plot_curvas_pr(curvas_dict, rep / "curvas_pr.png")

    plot_matrices_confusion(metricas_lista, rep / "matrices_confusion.png")
    plot_comparacion_barras(tabla, rep / "comparacion_barras.png")

    if benchmark_lista:
        plot_benchmark_entrenamiento(
            benchmark_lista, rep / "benchmark_entrenamiento.png"
        )
        with open(rep / "benchmark_entrenamiento.json", "w", encoding="utf-8") as f:
            json.dump(benchmark_lista, f, indent=2)

    logger.info("Reportes guardados en: %s", rep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    n_iter_busqueda: int = 12,
    n_jobs: int = -1,
    random_state: int = 42,
    test_size: float = 0.2,
) -> None:
    """Ejecuta el pipeline completo de entrenamiento SAPDE."""
    from sapde.config import configurar_logging, obtener_config
    configurar_logging()

    cfg   = obtener_config()
    rutas = cfg.rutas_absolutas()

    logger.info("=== SAPDE - Entrenamiento de modelos (OE2/OE3) ===")

    # 1. Cargar y preparar datos
    df = cargar_datos_entrenamiento(cfg)
    X, y = preparar_dataset_ml(df, incluir_features_avanzadas=True)

    logger.info(
        "Dataset ML: %d muestras x %d features | "
        "Desertores: %d (%.1f%%)",
        X.shape[0], X.shape[1], y.sum(), y.mean() * 100,
    )

    # 2. Split estratificado (OE3: semilla fija = reproducibilidad)
    X_arr = X.values
    y_arr = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr,
        test_size=test_size,
        stratify=y_arr,
        random_state=random_state,
    )
    logger.info(
        "Split: train=%d (%d positivos) | test=%d (%d positivos)",
        len(y_train), y_train.sum(),
        len(y_test),  y_test.sum(),
    )

    # 3. Entrenar todos los modelos
    metricas_lista, curvas_dict, benchmark_lista = entrenar_todos_los_modelos(
        X_train, y_train, X_test, y_test,
        cfg=cfg,
        n_iter_busqueda=n_iter_busqueda,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    # 4. Generar reportes
    generar_reportes(metricas_lista, curvas_dict, benchmark_lista, rutas)

    # 5. Resumen final
    tabla = tabla_comparacion(metricas_lista)
    mejor = tabla.iloc[0]
    logger.info(
        "Mejor modelo: %s | AUC=%.3f | Recall=%.3f | F1=%.3f",
        mejor["modelo"], mejor["auc_roc"], mejor["recall"], mejor["f1_score"],
    )

    print("\n" + "=" * 60)
    print("SAPDE — Resumen de modelos entrenados")
    print("=" * 60)
    print(tabla[["modelo", "precision", "recall", "f1_score", "auc_roc"]].to_string())
    print(f"\nMejor modelo: {mejor['modelo']} (AUC-ROC={mejor['auc_roc']:.3f})")
    print(f"Artefactos en: {rutas['models']}")
    print(f"Reportes en:   {rutas['reports']}")

    artefactos = listar_artefactos(rutas["models"])
    print(f"\nArtefactos guardados: {len(artefactos)}")


if __name__ == "__main__":
    main()
