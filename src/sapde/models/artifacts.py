"""
Gestión de artefactos versionados de modelos SAPDE (OE3).

Cada artefacto incluye:
  - pipeline entrenado (.joblib)
  - metadata completa (.json): métricas, parámetros, umbral, versión, timestamp
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger("sapde.models.artifacts")

_RAIZ = Path(__file__).resolve().parents[4]
_MODELS_DIR = _RAIZ / "models"


def _siguiente_version(nombre: str, directorio: Path | None = None) -> int:
    """Determina el número de versión siguiente para un modelo."""
    dir_busq = directorio or _MODELS_DIR
    dir_busq.mkdir(parents=True, exist_ok=True)
    existentes = list(dir_busq.glob(f"{nombre}_v*.joblib"))
    if not existentes:
        return 1
    versiones = []
    for p in existentes:
        try:
            v = int(p.stem.split("_v")[-1])
            versiones.append(v)
        except ValueError:
            pass
    return max(versiones) + 1 if versiones else 1


def guardar_pipeline(
    nombre: str,
    pipeline: Any,
    metricas: dict,
    umbral: float,
    mejores_params: dict | None = None,
    directorio: Path | None = None,
) -> Path:
    """
    Guarda el pipeline y su metadata en disco, versionados.

    Parámetros
    ----------
    nombre : str
        Nombre del modelo (ej. "RandomForest").
    pipeline : sklearn/imblearn Pipeline
        Pipeline entrenado y listo para producción.
    metricas : dict
        Resultados de evaluar_modelo().
    umbral : float
        Umbral de clasificación calibrado.
    mejores_params : dict, opcional
        Hiperparámetros encontrados por RandomizedSearchCV.
    directorio : Path, opcional
        Directorio destino. Por defecto: models/ en la raíz del proyecto.

    Retorna
    -------
    Path al archivo .joblib guardado.
    """
    dir_out = directorio or _MODELS_DIR
    dir_out.mkdir(parents=True, exist_ok=True)

    version = _siguiente_version(nombre, dir_out)
    nombre_archivo = f"{nombre}_v{version}"

    # Guardar pipeline con joblib (compresión para reducir tamaño)
    ruta_joblib = dir_out / f"{nombre_archivo}.joblib"
    joblib.dump(pipeline, ruta_joblib, compress=3)

    # Guardar metadata JSON
    metadata = {
        "nombre":       nombre,
        "version":      version,
        "timestamp":    datetime.now().isoformat(),
        "umbral":       umbral,
        "metricas":     metricas,
        "mejores_params": mejores_params or {},
        "archivo":      str(ruta_joblib),
    }
    ruta_json = dir_out / f"{nombre_archivo}_metadata.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Artefacto guardado: %s (AUC=%.3f, umbral=%.3f)",
        ruta_joblib, metricas.get("auc_roc", 0), umbral,
    )
    return ruta_joblib


def cargar_pipeline(nombre: str, version: int | None = None,
                    directorio: Path | None = None) -> tuple[Any, dict]:
    """
    Carga el pipeline y metadata de la versión especificada (o la última).

    Retorna
    -------
    (pipeline, metadata_dict)
    """
    dir_in = directorio or _MODELS_DIR

    if version is None:
        # Cargar la versión más reciente
        candidatos = list(dir_in.glob(f"{nombre}_v*.joblib"))
        if not candidatos:
            raise FileNotFoundError(
                f"No se encontró artefacto para '{nombre}' en {dir_in}"
            )
        candidatos.sort(key=lambda p: int(p.stem.split("_v")[-1]))
        ruta_joblib = candidatos[-1]
    else:
        ruta_joblib = dir_in / f"{nombre}_v{version}.joblib"
        if not ruta_joblib.exists():
            raise FileNotFoundError(f"Artefacto no encontrado: {ruta_joblib}")

    pipeline = joblib.load(ruta_joblib)

    ruta_json = ruta_joblib.with_name(ruta_joblib.stem + "_metadata.json")
    metadata  = {}
    if ruta_json.exists():
        with open(ruta_json, encoding="utf-8") as f:
            metadata = json.load(f)

    logger.info("Artefacto cargado: %s", ruta_joblib)
    return pipeline, metadata


def listar_artefactos(directorio: Path | None = None) -> list[dict]:
    """Lista todos los artefactos disponibles con sus métricas."""
    dir_in = directorio or _MODELS_DIR
    if not dir_in.exists():
        return []

    artefactos = []
    for ruta_json in sorted(dir_in.glob("*_metadata.json")):
        with open(ruta_json, encoding="utf-8") as f:
            meta = json.load(f)
        artefactos.append({
            "nombre":   meta.get("nombre"),
            "version":  meta.get("version"),
            "auc_roc":  meta.get("metricas", {}).get("auc_roc"),
            "recall":   meta.get("metricas", {}).get("recall"),
            "f1_score": meta.get("metricas", {}).get("f1_score"),
            "umbral":   meta.get("umbral"),
            "timestamp":meta.get("timestamp"),
        })
    return artefactos


def mejor_modelo(directorio: Path | None = None) -> tuple[Any, dict]:
    """
    Carga el artefacto con el mayor AUC-ROC entre todos los disponibles.
    """
    artefactos = listar_artefactos(directorio)
    if not artefactos:
        raise FileNotFoundError("No hay artefactos entrenados.")
    mejor = max(artefactos, key=lambda x: x.get("auc_roc") or 0)
    logger.info(
        "Mejor modelo: %s v%s (AUC=%.3f)",
        mejor["nombre"], mejor["version"], mejor["auc_roc"] or 0,
    )
    return cargar_pipeline(mejor["nombre"], mejor["version"], directorio)
