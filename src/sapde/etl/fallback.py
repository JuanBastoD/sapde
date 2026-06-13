"""
ETL paralelo en Python — fallback para cuando C++/OpenMP no está disponible.

Implementa las mismas operaciones que etl_main.cpp usando joblib para
paralelizar sobre chunks del DataFrame.

OE1: Limpieza, normalización y feature engineering paralelizados.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger("sapde.etl.fallback")


@dataclass
class ResultadoETL:
    hilos: int
    registros: int
    t_lectura: float
    t_etl: float
    t_escritura: float
    t_total: float
    modo: str = "python"


# ── Operaciones atómicas sobre un chunk ──────────────────────────────────────

def _limpiar_chunk(df: pd.DataFrame, lim_reprobadas: float) -> pd.DataFrame:
    """
    Paso 1 y 2: clip por rangos de dominio + IQR para asignaturas reprobadas.
    Se ejecuta en paralelo sobre chunks independientes del DataFrame.
    """
    df = df.copy()

    # Clip de rangos conocidos (escala colombiana 0-5)
    df["promedio_semestre"]  = df["promedio_semestre"].clip(1.0, 5.0)
    df["promedio_acumulado"] = df["promedio_acumulado"].clip(1.0, 5.0)
    df["tasa_reprobacion"]   = df["tasa_reprobacion"].clip(0.0, 1.0)
    df["porcentaje_avance_real"] = df["porcentaje_avance_real"].clip(0.0, 100.0)

    # Créditos aprobados ≤ matriculados
    df["creditos_aprobados"] = df[["creditos_aprobados", "creditos_matriculados"]].min(axis=1)
    df["creditos_aprobados"] = df["creditos_aprobados"].clip(lower=0)

    df["interrupciones_previas"] = df["interrupciones_previas"].clip(lower=0)

    # Clip IQR sobre reprobadas (límite calculado globalmente y pasado como parámetro)
    df["num_asignaturas_reprobadas"] = df["num_asignaturas_reprobadas"].clip(
        upper=int(lim_reprobadas)
    )

    return df


def _transformar_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pasos 3 y 4: decodificación de cohorte + feature engineering.
    """
    df = df.copy()

    # Decodificar cohorte "YYYY-I" → año (int) + semestre (1 o 2)
    partes = df["cohorte"].str.extract(r"^(\d{4})-(I{1,2})$")
    df["anio_cohorte"]     = pd.to_numeric(partes[0], errors="coerce").fillna(0).astype(int)
    df["semestre_cohorte"] = partes[1].map({"I": 1, "II": 2}).fillna(0).astype(int)

    # ratio_aprobacion: fracción de créditos superados [0, 1]
    df["ratio_aprobacion"] = np.where(
        df["creditos_matriculados"] > 0,
        df["creditos_aprobados"] / df["creditos_matriculados"],
        0.0,
    )

    # deficit_creditos: créditos no aprobados (rezago curricular)
    df["deficit_creditos"] = df["creditos_matriculados"] - df["creditos_aprobados"]

    # score_riesgo_bruto ∈ [0, 1]: mayor → mayor riesgo
    f_nota   = (5.0 - df["promedio_acumulado"]) / 4.0
    f_repro  = df["tasa_reprobacion"]
    f_avance = 1.0 - df["porcentaje_avance_real"] / 100.0
    f_interr = (df["interrupciones_previas"] / 2.0).clip(upper=1.0)

    df["score_riesgo_bruto"] = (
        0.40 * f_nota   +
        0.30 * f_repro  +
        0.20 * f_avance +
        0.10 * f_interr
    )

    return df


# ── Pipeline principal ────────────────────────────────────────────────────────

def etl_python(
    df_raw: pd.DataFrame,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Aplica el ETL completo en paralelo usando joblib.

    Parámetros
    ----------
    df_raw : DataFrame con el CSV crudo (columnas del esquema SAPDE).
    n_jobs : número de workers (-1 = todos los núcleos disponibles).

    Retorna
    -------
    DataFrame procesado con columnas originales + derivadas.
    """
    n = len(df_raw)
    # joblib no lanza workers para datasets pequeños
    n_workers = n_jobs if n_jobs > 0 else 4
    n_chunks  = max(1, n_workers)

    logger.info("ETL Python: %d registros, n_jobs=%d, chunks=%d", n, n_jobs, n_chunks)

    # Límite IQR global (debe calcularse antes de dividir en chunks)
    q1 = df_raw["num_asignaturas_reprobadas"].quantile(0.25)
    q3 = df_raw["num_asignaturas_reprobadas"].quantile(0.75)
    lim_reprobadas = q3 + 1.5 * (q3 - q1)

    chunks = np.array_split(df_raw, n_chunks)

    # ── Paso 1-2: limpieza en paralelo ──────────────────────────────────────
    chunks_limpios = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_limpiar_chunk)(ch, lim_reprobadas) for ch in chunks
    )

    # ── Pasos 3-4: transformación + features en paralelo ───────────────────
    chunks_finales = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_transformar_chunk)(ch) for ch in chunks_limpios
    )

    df_out = pd.concat(chunks_finales, ignore_index=True)
    logger.info("ETL Python completado: %d registros -> %d columnas", len(df_out), len(df_out.columns))
    return df_out


def ejecutar_etl_python(
    input_csv: "Path",
    output_csv: "Path",
    n_jobs: int = 1,
) -> ResultadoETL:
    """
    Carga el CSV, aplica el ETL Python en paralelo y guarda el resultado.

    Devuelve ResultadoETL con tiempos medidos (compatible con cpp_runner).
    """
    from pathlib import Path

    t0 = time.perf_counter()

    # Lectura
    t_lec_ini = time.perf_counter()
    df_raw = pd.read_csv(input_csv, encoding="utf-8")
    t_lectura = time.perf_counter() - t_lec_ini

    # ETL
    t_etl_ini = time.perf_counter()
    df_out = etl_python(df_raw, n_jobs=n_jobs)
    t_etl = time.perf_counter() - t_etl_ini

    # Escritura
    t_esc_ini = time.perf_counter()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    t_escritura = time.perf_counter() - t_esc_ini

    t_total = time.perf_counter() - t0

    return ResultadoETL(
        hilos=n_jobs if n_jobs > 0 else 4,
        registros=len(df_out),
        t_lectura=t_lectura,
        t_etl=t_etl,
        t_escritura=t_escritura,
        t_total=t_total,
        modo="python",
    )
