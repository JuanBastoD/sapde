"""
Orquestador de explicabilidad SHAP para SAPDE (OE4).

Flujo:
  1. Carga datos procesados y split estratificado (misma semilla que train.py)
  2. Carga artefactos de modelos entrenados (Fase 2)
  3. Para cada modelo calcula SHAP global (importancia de features)
  4. Genera plots: summary beeswarm, summary bar, waterfall por estudiante
  5. Guarda CSV de importancias y arrays SHAP en reports/shap/
  6. Genera comparativa de importancias entre modelos

Uso:
    python -m sapde.explain.run_shap
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sapde.config import configurar_logging, obtener_config
from sapde.explain.shap_explainer import ExplicadorSHAP
from sapde.features.engineering import preparar_dataset_ml
from sapde.models.artifacts import cargar_pipeline, listar_artefactos
from sapde.models.train import cargar_datos_entrenamiento

logger = logging.getLogger("sapde.explain.run_shap")

# Semilla y split idénticos a train.py para reproducibilidad
_SEMILLA = 42
_TEST_SIZE = 0.2

# Número de muestras para SHAP global (balance velocidad/precisión)
_N_GLOBAL = 300
# Número de estudiantes de ejemplo para waterfall
_N_WATERFALL = 3


def _nombres_disponibles(directorio: Path) -> list[str]:
    """Extrae los nombres únicos de modelos disponibles en el directorio."""
    artefactos = listar_artefactos(directorio)
    vistos: dict[str, float] = {}
    for a in artefactos:
        n = a["nombre"]
        auc = a.get("auc_roc") or 0.0
        if n not in vistos or auc > vistos[n]:
            vistos[n] = auc
    return list(vistos.keys())


def _plot_comparativa_modelos(
    importancias: dict[str, pd.DataFrame],
    ruta_salida: Path,
    top_n: int = 10,
) -> None:
    """Heatmap de importancia media |SHAP| top-N features × modelos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Unificar features: top_n globales por rank promedio
    todos_features: dict[str, list[float]] = {}
    for nombre, df in importancias.items():
        for _, row in df.iterrows():
            feat = row["feature"]
            if feat not in todos_features:
                todos_features[feat] = []
            todos_features[feat].append(row["importancia_media"])

    rank_promedio = {
        feat: np.mean(vals) for feat, vals in todos_features.items()
    }
    top_features = sorted(rank_promedio, key=rank_promedio.get, reverse=True)[:top_n]

    # Construir matriz: features × modelos
    modelos = list(importancias.keys())
    matriz = np.zeros((len(top_features), len(modelos)))
    for j, nombre in enumerate(modelos):
        df = importancias[nombre].set_index("feature")
        for i, feat in enumerate(top_features):
            if feat in df.index:
                matriz[i, j] = df.loc[feat, "importancia_media"]

    # Normalizar por columna para comparación visual
    col_max = matriz.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    matriz_norm = matriz / col_max

    fig, ax = plt.subplots(figsize=(max(8, len(modelos) * 1.6), max(6, len(top_features) * 0.55)))
    im = ax.imshow(matriz_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Importancia relativa (max=1)")
    ax.set_xticks(range(len(modelos)))
    ax.set_xticklabels(modelos, fontsize=10)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_title(f"Importancia SHAP relativa — Top {top_n} features", fontsize=13, pad=10)
    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=120, bbox_inches="tight")
    plt.close("all")
    logger.info("Comparativa SHAP guardada: %s", ruta_salida)


def main() -> None:
    configurar_logging()
    cfg = obtener_config()
    rutas = cfg.rutas_absolutas()
    dir_shap = rutas["reports"] / "shap"
    dir_shap.mkdir(parents=True, exist_ok=True)

    logger.info("=== SAPDE — Interpretabilidad SHAP (OE4) ===")

    # ── 1. Datos ─────────────────────────────────────────────────────────────
    df = cargar_datos_entrenamiento(cfg)
    X, y = preparar_dataset_ml(df, incluir_features_avanzadas=True)
    feature_names = X.columns.tolist()

    X_arr = X.values
    y_arr = y.values
    _, X_test, _, y_test = train_test_split(
        X_arr, y_arr,
        test_size=_TEST_SIZE,
        stratify=y_arr,
        random_state=_SEMILLA,
    )

    logger.info(
        "Datos de test: %d muestras | %d positivos (%.1f%%)",
        len(y_test), y_test.sum(), y_test.mean() * 100,
    )

    # Índices de desertores confirmados (para waterfall ilustrativo)
    idx_desertores = np.where(y_test == 1)[0]
    idx_permanentes = np.where(y_test == 0)[0]

    # ── 2. Cargar artefactos ─────────────────────────────────────────────────
    nombres = _nombres_disponibles(rutas["models"])
    if not nombres:
        raise RuntimeError(
            "No hay artefactos de modelos. Ejecuta primero: "
            "python -m sapde.models.train"
        )
    logger.info("Modelos a explicar: %s", nombres)

    importancias: dict[str, pd.DataFrame] = {}

    for nombre in nombres:
        logger.info("--- Procesando SHAP: %s ---", nombre)

        try:
            pipeline, metadata = cargar_pipeline(nombre, directorio=rutas["models"])
        except FileNotFoundError:
            logger.warning("No se encontro artefacto para %s. Saltando.", nombre)
            continue

        umbral = metadata.get("umbral", 0.5)

        # Crear explainer
        explainer = ExplicadorSHAP(
            nombre=nombre,
            pipeline=pipeline,
            feature_names=feature_names,
            X_background=X_test,
            n_background=200,
        )

        # ── Importancia global ────────────────────────────────────────────
        logger.info("  Calculando importancia global (%d muestras)...", _N_GLOBAL)
        df_imp = explainer.importancia_global(X_test, n_muestras=_N_GLOBAL)
        importancias[nombre] = df_imp

        ruta_imp = dir_shap / f"{nombre}_importancia.csv"
        df_imp.to_csv(ruta_imp, index=False, encoding="utf-8")
        logger.info("  Top 5 features: %s", df_imp["feature"].head(5).tolist())

        # ── Plots globales ────────────────────────────────────────────────
        explainer.plot_summary(
            X_test, dir_shap / f"{nombre}_summary.png", n_muestras=_N_GLOBAL
        )
        explainer.plot_bar(
            X_test, dir_shap / f"{nombre}_bar.png", n_muestras=_N_GLOBAL
        )
        explainer.plot_importancia_comparativa(
            df_imp, dir_shap / f"{nombre}_importancia.png"
        )

        # ── Waterfall por estudiante ─────────────────────────────────────
        for i_sample, tipo in [
            (idx_desertores[0], "desertor"),
            (idx_permanentes[0], "permanente"),
        ]:
            x_single = X_test[i_sample]
            proba = float(pipeline.predict_proba(x_single.reshape(1, -1))[0, 1])
            label = f"{tipo} | P={proba:.3f} | umbral={umbral:.3f}"
            explainer.plot_waterfall(
                x_single,
                dir_shap / f"{nombre}_waterfall_{tipo}.png",
                label_estudiante=label,
            )

    # ── 3. Comparativa entre modelos ─────────────────────────────────────────
    if len(importancias) > 1:
        _plot_comparativa_modelos(
            importancias, dir_shap / "comparativa_modelos_shap.png", top_n=12
        )

    # ── 4. Resumen en JSON ────────────────────────────────────────────────────
    resumen = {}
    for nombre, df_imp in importancias.items():
        resumen[nombre] = {
            "top_features": df_imp["feature"].head(10).tolist(),
            "importancias": df_imp.head(10).set_index("feature")["importancia_media"].to_dict(),
        }

    with open(dir_shap / "shap_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    logger.info("=== SHAP completado. Reportes en: %s ===", dir_shap)
    print("\n" + "=" * 60)
    print("SAPDE — Interpretabilidad SHAP completada")
    print("=" * 60)
    for nombre, df_imp in importancias.items():
        top5 = df_imp["feature"].head(5).tolist()
        print(f"  {nombre}: {top5}")
    print(f"\nReportes en: {dir_shap}")


if __name__ == "__main__":
    main()
