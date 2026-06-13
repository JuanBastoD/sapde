"""
Ingeniería de variables para el modelo de predicción de deserción.

Transforma el dataset procesado por el ETL en el conjunto final de features
listo para el pipeline ML (Fase 2).

Las variables generadas aquí son adicionales a las del ETL C++.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("sapde.features.engineering")

# Columnas numéricas base que entran al modelo
FEATURES_NUMERICAS = [
    "promedio_semestre",
    "promedio_acumulado",
    "tendencia_calificaciones",
    "creditos_matriculados",
    "creditos_aprobados",
    "porcentaje_avance_real",
    "num_asignaturas_reprobadas",
    "tasa_reprobacion",
    "semestres_cursados",
    "interrupciones_previas",
    # Derivadas del ETL
    "ratio_aprobacion",
    "deficit_creditos",
    "score_riesgo_bruto",
    "anio_cohorte",
    "semestre_cohorte",
]

COLUMNA_OBJETIVO = "deserta"


def agregar_features_avanzadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables derivadas de segundo nivel al DataFrame procesado.

    Operaciones:
    - Interacciones entre variables de rendimiento y avance
    - Indicadores binarios de umbrales críticos
    - Semestre al cuadrado (captura no-linealidad)
    - Relación entre tendencia y tasa de reprobación

    El DataFrame debe haber pasado previamente por el ETL (contiene ratio_aprobacion, etc.)
    """
    df = df.copy()

    # ── Interacciones ───────────────────────────────────────────────────────
    # Nota × avance: estudiante con buenas notas Y mucho avance = bajo riesgo
    df["nota_x_avance"] = df["promedio_acumulado"] * df["porcentaje_avance_real"] / 100.0

    # Reprobación × déficit: alguien que reprueba mucho Y tiene déficit acumulado
    df["repro_x_deficit"] = df["tasa_reprobacion"] * df["deficit_creditos"]

    # ── Indicadores binarios de umbral crítico ───────────────────────────────
    # Nota por debajo del umbral de riesgo alto (< 3.2 ≈ promedio bajo)
    df["flag_nota_critica"]   = (df["promedio_acumulado"] < 3.2).astype(int)
    # Más de 1 asignatura reprobada este semestre
    df["flag_reprobacion"]    = (df["num_asignaturas_reprobadas"] > 1).astype(int)
    # Avance inferior al 50% del programa
    df["flag_rezago"]         = (df["porcentaje_avance_real"] < 50.0).astype(int)
    # Tendencia negativa en notas
    df["flag_tendencia_neg"]  = (df["tendencia_calificaciones"] < -0.05).astype(int)
    # Tiene interrupciones previas
    df["flag_interrupcion"]   = (df["interrupciones_previas"] > 0).astype(int)

    # ── No-linealidades ─────────────────────────────────────────────────────
    df["semestre_sq"]  = df["semestre"] ** 2

    # ── Combinación de flags → riesgo discreto (0-5) ────────────────────────
    flags = [
        "flag_nota_critica", "flag_reprobacion", "flag_rezago",
        "flag_tendencia_neg", "flag_interrupcion",
    ]
    df["n_factores_riesgo"] = df[flags].sum(axis=1)

    logger.debug(
        "Features avanzadas agregadas: %d nuevas columnas",
        len(["nota_x_avance", "repro_x_deficit"] + flags + ["semestre_sq", "n_factores_riesgo"]),
    )
    return df


def obtener_lista_features(df: pd.DataFrame) -> list[str]:
    """
    Retorna la lista final de features numéricas disponibles en el DataFrame.
    Excluye columnas de identificación y la variable objetivo.
    """
    excluir = {"student_id", "cohorte", COLUMNA_OBJETIVO}
    return [c for c in df.columns if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]


def preparar_dataset_ml(
    df_etl: pd.DataFrame,
    incluir_features_avanzadas: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara (X, y) listo para el pipeline ML de scikit-learn.

    Parámetros
    ----------
    df_etl : DataFrame de salida del ETL.
    incluir_features_avanzadas : si True, agrega las features de segundo nivel.

    Retorna
    -------
    X : DataFrame de features (float64)
    y : Series binaria (0/1) — variable objetivo
    """
    if incluir_features_avanzadas:
        df = agregar_features_avanzadas(df_etl)
    else:
        df = df_etl.copy()

    features = obtener_lista_features(df)
    X = df[features].astype(np.float64)
    y = df[COLUMNA_OBJETIVO].astype(int)

    # Verificar que no hay nulos
    nulos = X.isnull().sum().sum()
    if nulos > 0:
        logger.warning("%d valores nulos en X; rellenando con mediana.", nulos)
        X = X.fillna(X.median())

    logger.info(
        "Dataset ML listo: %d muestras, %d features, %d positivos (%.1f%%)",
        len(X), len(features), y.sum(), y.mean() * 100,
    )
    return X, y


def guardar_dataset_procesado(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda el dataset completo (con features) como Parquet."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    logger.info("Dataset procesado guardado: %s (%d filas, %d cols)", ruta, len(df), len(df.columns))


def main() -> None:
    """CLI: carga el ETL procesado, agrega features y guarda Parquet final."""
    from sapde.config import obtener_config, configurar_logging
    configurar_logging()

    cfg   = obtener_config()
    rutas = cfg.rutas_absolutas()

    etl_csv  = rutas["interim"] / "estudiantes_etl.csv"
    if not etl_csv.exists():
        logger.error("ETL no ejecutado. Corre primero: python -m sapde.etl.pipeline")
        return

    df_etl  = pd.read_csv(etl_csv, encoding="utf-8")
    df_feat = agregar_features_avanzadas(df_etl)

    ruta_parquet = rutas["processed"] / "dataset_final.parquet"
    guardar_dataset_procesado(df_feat, ruta_parquet)

    X, y = preparar_dataset_ml(df_feat)
    print(f"\nDataset final: {X.shape[0]} muestras × {X.shape[1]} features")
    print(f"Desertores: {y.sum()} ({y.mean():.1%})")
    print("\nFeatures disponibles:")
    for f in obtener_lista_features(df_feat):
        print(f"  {f}")


if __name__ == "__main__":
    main()
