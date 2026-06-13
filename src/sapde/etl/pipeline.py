"""
Pipeline ETL de SAPDE (OE1).

Orquestador principal:
  1. Intenta compilar y ejecutar el ETL C++/OpenMP.
  2. Si C++ no está disponible, activa el fallback Python (joblib).
  3. Ejecuta el benchmark con 1, 2, 4 y 8 hilos/workers.
  4. Guarda los resultados de tiempo en reports/benchmark_etl.json.

Uso directo:
    python -m sapde.etl.pipeline
"""

import json
import logging
import time
from pathlib import Path
from typing import Literal

import pandas as pd

from sapde.config import obtener_config
from sapde.etl.cpp_runner import CppNoDisponible, ResultadoETL
from sapde.etl.fallback import ejecutar_etl_python

logger = logging.getLogger("sapde.etl.pipeline")

ModoETL = Literal["cpp", "python", "auto"]


def _guardar_benchmark(resultados: list[dict], ruta: Path) -> None:
    """Guarda los tiempos del benchmark en formato JSON."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark guardado en: %s", ruta)


def ejecutar_etl(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    modo: ModoETL = "auto",
    n_hilos: int = 1,
    benchmark: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline ETL completo.

    Parámetros
    ----------
    input_csv  : CSV crudo de entrada (por defecto: data/raw/estudiantes_sintetico.csv)
    output_csv : CSV procesado de salida (por defecto: data/interim/estudiantes_etl.csv)
    modo       : "cpp" | "python" | "auto" (auto prueba C++ y cae a Python si falla)
    n_hilos    : número de hilos/workers para la ejecución
    benchmark  : si True, ejecuta con 1, 2, 4, 8 hilos y guarda tiempos

    Retorna
    -------
    DataFrame con el dataset procesado.
    """
    cfg   = obtener_config()
    rutas = cfg.rutas_absolutas()

    input_csv  = input_csv  or rutas["raw"] / "estudiantes_sintetico.csv"
    output_csv = output_csv or rutas["interim"] / "estudiantes_etl.csv"

    if not input_csv.exists():
        raise FileNotFoundError(
            f"CSV de entrada no encontrado: {input_csv}\n"
            "Ejecuta primero: python -m sapde.data.synthetic"
        )

    # ── Selección de modo ─────────────────────────────────────────────────────
    usar_cpp = False
    exe_path = None

    if modo in ("cpp", "auto"):
        try:
            from sapde.etl.cpp_runner import compilar, ejecutar
            exe_path = compilar()
            usar_cpp = True
            logger.info("Modo ETL: C++/OpenMP")
        except CppNoDisponible as e:
            if modo == "cpp":
                raise
            logger.warning("C++ no disponible (%s). Usando fallback Python.", e)

    if not usar_cpp:
        logger.info("Modo ETL: Python (joblib)")

    # ── Ejecución simple (sin benchmark) ─────────────────────────────────────
    if not benchmark:
        if usar_cpp:
            from sapde.etl.cpp_runner import ejecutar
            res = ejecutar(exe_path, input_csv, output_csv, n_hilos)
        else:
            res = ejecutar_etl_python(input_csv, output_csv, n_jobs=n_hilos)

        logger.info(
            "ETL (%s): %d registros en %.3fs (ETL puro: %.3fs)",
            res.modo, res.registros, res.t_total, res.t_etl,
        )
        return pd.read_csv(output_csv, encoding="utf-8")

    # ── Benchmark: 1, 2, 4, 8 hilos ──────────────────────────────────────────
    logger.info("Iniciando benchmark ETL (1, 2, 4, 8 hilos)...")
    registros_benchmark: list[dict] = []
    t1_referencia = None

    for n in [1, 2, 4, 8]:
        logger.info("  Ejecutando con %d hilo(s)...", n)
        out_tmp = rutas["interim"] / f"etl_bench_{n}hilos.csv"

        if usar_cpp:
            from sapde.etl.cpp_runner import ejecutar
            res = ejecutar(exe_path, input_csv, out_tmp, n)
        else:
            res = ejecutar_etl_python(input_csv, out_tmp, n_jobs=n)

        t_etl = res.t_etl
        if n == 1:
            t1_referencia = t_etl

        speedup    = t1_referencia / t_etl if t_etl > 0 else 0.0
        eficiencia = speedup / n

        entrada = {
            "hilos":      n,
            "modo":       res.modo,
            "registros":  res.registros,
            "t_etl_s":    round(t_etl, 4),
            "t_total_s":  round(res.t_total, 4),
            "speedup":    round(speedup, 3),
            "eficiencia": round(eficiencia, 3),
        }
        registros_benchmark.append(entrada)
        logger.info(
            "  %d hilo(s): ETL=%.4fs  Speedup=%.2fx  Eficiencia=%.1f%%",
            n, t_etl, speedup, eficiencia * 100,
        )

    # Guardar benchmark
    ruta_bench = rutas["reports"] / "benchmark_etl.json"
    _guardar_benchmark(registros_benchmark, ruta_bench)

    # La ejecución final (8 hilos) ya dejó output_csv; renombrar si es necesario
    out_final = rutas["interim"] / "etl_bench_8hilos.csv"
    if out_final.exists() and not output_csv.exists():
        import shutil
        shutil.copy(out_final, output_csv)

    # Usar el resultado de 1 hilo como dataset de referencia
    out_1hilo = rutas["interim"] / "etl_bench_1hilos.csv"
    if out_1hilo.exists():
        import shutil
        shutil.copy(out_1hilo, output_csv)

    logger.info("Benchmark completado. Resultados en %s", ruta_bench)
    for r in registros_benchmark:
        logger.info(
            "  %d hilos | t_etl=%.4fs | speedup=%.2fx | eficiencia=%.1f%%",
            r["hilos"], r["t_etl_s"], r["speedup"], r["eficiencia"] * 100,
        )

    return pd.read_csv(output_csv, encoding="utf-8")


def main() -> None:
    """Punto de entrada CLI: ejecuta ETL + benchmark."""
    from sapde.config import configurar_logging
    configurar_logging()

    cfg   = obtener_config()
    rutas = cfg.rutas_absolutas()

    logger.info("=== Pipeline ETL SAPDE ===")
    df = ejecutar_etl(
        modo="auto",
        n_hilos=1,
        benchmark=True,
    )

    # Convertir a Parquet para el pipeline ML
    parquet_out = rutas["processed"] / "dataset_etl.parquet"
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_out, index=False)
    logger.info("Parquet guardado: %s (%d filas, %d cols)", parquet_out, len(df), len(df.columns))

    print(f"\nETL completado: {len(df)} registros, {len(df.columns)} columnas")
    print(f"Parquet: {parquet_out}")
    print(df.describe(include="all").round(3).to_string())


if __name__ == "__main__":
    main()
