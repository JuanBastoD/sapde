"""
Orquestador de validación y generación de reporte — SAPDE Fase 6.

Flujo:
  1. Carga el dataset y reproduce el split train/test (semilla=42)
  2. Determina los modelos únicos más recientes (mejor versión por nombre)
  3. Ejecuta validación completa: bootstrap IC + CV 5-fold + calibración
  4. Genera gráficas de speedup y el reporte HTML autocontenido

Uso:
    python -m sapde.validation.run_validation
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("sapde.validation.run")

_RAIZ       = Path(__file__).resolve().parents[3]
_MODELS_DIR = _RAIZ / "models"
_REPORTS    = _RAIZ / "reports"
_SEMILLA    = 42
_TEST_SIZE  = 0.2
_N_BOOTSTRAP = 500   # 500 iter ≈ 30 s; subir a 1000 para publicación


def _cargar_split():
    from sapde.etl.pipeline import ejecutar_etl
    from sapde.features.engineering import preparar_dataset_ml

    csv = _RAIZ / "data" / "raw" / "estudiantes_sintetico.csv"
    df_etl = ejecutar_etl(input_csv=csv, modo="python", n_hilos=1)
    X_df, y = preparar_dataset_ml(df_etl, incluir_features_avanzadas=True)
    X = X_df.values.astype(np.float64)
    y = y.values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=_SEMILLA, stratify=y
    )
    logger.info(
        "Dataset: %d train | %d test | %.1f%% positivos",
        len(y_train), len(y_test), y.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def _modelos_a_validar() -> list[dict]:
    """Una entrada por nombre de modelo (versión más alta disponible)."""
    from sapde.models.artifacts import listar_artefactos
    artefactos = listar_artefactos(_MODELS_DIR)
    if not artefactos:
        raise RuntimeError("No hay artefactos. Ejecuta primero: python -m sapde.models.train")

    # Mejor versión por nombre
    por_nombre: dict[str, dict] = {}
    for art in artefactos:
        nombre = art["nombre"]
        if nombre not in por_nombre or art["version"] > por_nombre[nombre]["version"]:
            por_nombre[nombre] = art

    return list(por_nombre.values())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    from sapde.validation.validador import evaluar_modelos_completo
    from sapde.validation.report_generator import generar_html
    from sapde.dashboard.utils import cargar_shap_resumen

    # 1. Datos
    logger.info("=== SAPDE Fase 6 — Validación y Benchmark ===")
    X_train, X_test, y_train, y_test = _cargar_split()

    # 2. Modelos a validar
    artefactos = _modelos_a_validar()
    logger.info("Modelos a validar: %s", [a["nombre"] for a in artefactos])

    # 3. Validación estadística
    resultados = evaluar_modelos_completo(
        artefactos, X_train, y_train, X_test, y_test,
        n_splits=5, n_bootstrap=_N_BOOTSTRAP, semilla=_SEMILLA,
    )

    # 4. Guardar resultados JSON
    ruta_json = _REPORTS / "validacion" / "resultados_validacion.json"
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    logger.info("Resultados JSON: %s", ruta_json)

    # 5. Leer benchmarks existentes
    benchmark_etl   = _leer_json(_REPORTS / "benchmark_etl.json")
    benchmark_train = _leer_json(_REPORTS / "benchmark_entrenamiento.json")

    # 6. SHAP resumen
    shap_resumen = cargar_shap_resumen()

    # 7. Reporte HTML
    ruta_html = _REPORTS / "validacion" / "reporte_validacion.html"
    generar_html(resultados, benchmark_etl, benchmark_train, ruta_html, shap_resumen)

    # Resumen por consola
    _imprimir_resumen(resultados)

    logger.info("=== Fase 6 completada ===")
    logger.info("Reporte: %s", ruta_html)
    return resultados


def _leer_json(ruta: Path) -> list[dict]:
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    logger.warning("No encontrado: %s", ruta)
    return []


def _imprimir_resumen(resultados: list[dict]):
    print("\n" + "=" * 70)
    print(f"{'Modelo':<15} {'AUC-ROC':>10} {'IC95% Lower':>12} {'IC95% Upper':>12} {'Recall':>9} {'ECE':>7}")
    print("-" * 70)
    for r in sorted(resultados, key=lambda x: -x["bootstrap"]["auc_roc"]["media"]):
        bs = r["bootstrap"]
        print(
            f"{r['nombre']:<15} "
            f"{bs['auc_roc']['media']:>10.4f} "
            f"{bs['auc_roc']['lower']:>12.4f} "
            f"{bs['auc_roc']['upper']:>12.4f} "
            f"{bs['recall']['media']:>9.4f} "
            f"{r['calibracion']['ece']:>7.4f}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
