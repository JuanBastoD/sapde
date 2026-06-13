"""
Utilidades compartidas del dashboard SAPDE.

Carga el modelo, el explainer SHAP y la sesión DB una sola vez
usando @st.cache_resource para persistir entre reruns.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_RAIZ = Path(__file__).resolve().parents[3]
_MODELS_DIR = _RAIZ / "models"
_SHAP_REPORT = _RAIZ / "reports" / "shap" / "shap_resumen.json"

# Orden canónico de las 25 features — debe coincidir con el orden de columnas
# producido por etl_python() + agregar_features_avanzadas() (obtenido de preparar_dataset_ml)
FEATURE_NAMES: list[str] = [
    "semestre", "promedio_semestre", "promedio_acumulado",
    "tendencia_calificaciones", "creditos_matriculados", "creditos_aprobados",
    "porcentaje_avance_real", "num_asignaturas_reprobadas", "tasa_reprobacion",
    "semestres_cursados", "interrupciones_previas",
    "anio_cohorte", "semestre_cohorte",          # el ETL las pone antes de ratio/deficit
    "ratio_aprobacion", "deficit_creditos", "score_riesgo_bruto",
    "nota_x_avance", "repro_x_deficit", "flag_nota_critica", "flag_reprobacion",
    "flag_rezago", "flag_tendencia_neg", "flag_interrupcion",
    "semestre_sq", "n_factores_riesgo",
]

NIVELES_COLOR = {
    "muy_alto": "#C62828",
    "alto":     "#E65100",
    "moderado": "#B8860B",
    "bajo":     "#2E7D32",
}

NIVEL_ETIQUETA = {
    "muy_alto": "Muy Alto",
    "alto":     "Alto",
    "moderado": "Moderado",
    "bajo":     "Bajo",
}


# ── Carga de recursos ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    from sapde.models.artifacts import mejor_modelo
    pipeline, raw_meta = mejor_modelo(_MODELS_DIR)

    scaler = pipeline.named_steps["scaler"]
    fn = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else FEATURE_NAMES

    meta = {
        "nombre":   raw_meta.get("nombre", ""),
        "version":  f"v{raw_meta.get('version', '?')}",
        "auc_roc":  raw_meta.get("metricas", {}).get("auc_roc", 0.0),
        "recall":   raw_meta.get("metricas", {}).get("recall", 0.0),
        "f1_score": raw_meta.get("metricas", {}).get("f1_score", 0.0),
        "umbral":   raw_meta.get("umbral", 0.5),
        "timestamp": raw_meta.get("timestamp", ""),
    }
    return pipeline, meta, fn


@st.cache_resource(show_spinner="Inicializando SHAP...")
def cargar_explainer(_pipeline, feature_names: list[str]):
    from sapde.explain.shap_explainer import ExplicadorSHAP
    from sapde.etl.fallback import etl_python
    from sapde.features.engineering import preparar_dataset_ml

    csv = _RAIZ / "data" / "raw" / "estudiantes_sintetico.csv"
    df_raw = pd.read_csv(csv)
    df_etl = etl_python(df_raw, n_jobs=1)
    X, _ = preparar_dataset_ml(df_etl, incluir_features_avanzadas=True)
    X = X[feature_names]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(300, len(X)), replace=False)
    X_bg = X.values[idx]

    meta_nombre = _pipeline.named_steps["modelo"].__class__.__name__
    return ExplicadorSHAP(
        nombre=meta_nombre,
        pipeline=_pipeline,
        feature_names=feature_names,
        X_background=X_bg,
        n_background=200,
    )


@st.cache_resource(show_spinner="Conectando a base de datos...")
def obtener_engine():
    from sapde.db.session import engine, init_db
    init_db()
    return engine


# ── Predicción individual ─────────────────────────────────────────────────────

def nivel_riesgo(proba: float, umbral: float) -> str:
    if proba < 0.20:
        return "bajo"
    if proba < umbral:
        return "moderado"
    if proba < 0.70:
        return "alto"
    return "muy_alto"


def entrada_a_features(datos: dict, fn: list[str]) -> np.ndarray:
    from sapde.features.engineering import agregar_features_avanzadas
    row = {**datos, "student_id": "dashboard", "deserta": 0}
    df = pd.DataFrame([row])
    df = agregar_features_avanzadas(df)
    return df[fn].values.astype(float)


def predecir(datos: dict, pipeline, meta: dict, fn: list[str]) -> dict:
    X = entrada_a_features(datos, fn)
    proba = float(pipeline.predict_proba(X)[0, 1])
    umbral = meta["umbral"]
    nivel = nivel_riesgo(proba, umbral)
    return {
        "probabilidad": proba,
        "prediccion":   proba >= umbral,
        "nivel_riesgo": nivel,
        "umbral":       umbral,
        "modelo":       meta["nombre"],
        "version":      meta["version"],
        "X":            X[0],
    }


def explicar(X_raw: np.ndarray, explainer) -> dict:
    return explainer.explicar_estudiante(X_raw)


# ── Persistencia DB ───────────────────────────────────────────────────────────

def guardar_prediccion_db(engine, student_id: str, semestre: int,
                          resultado: dict, features: dict,
                          shap_exp: Optional[dict] = None):
    import json
    from sqlalchemy.orm import sessionmaker
    from sapde.db.crud import crear_prediccion

    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        shap_data = None
        if shap_exp:
            sv = shap_exp["por_feature"]
            sorted_sv = sorted(sv.items(), key=lambda kv: kv[1])
            shap_data = {
                "base_value": shap_exp["base_value"],
                "feature_contributions": sv,
                "top_factores_riesgo": [k for k, v in sorted_sv if v > 0][-3:][::-1],
                "top_factores_protectores": [k for k, v in sorted_sv if v < 0][:3],
            }
        crear_prediccion(
            db=db,
            student_id=student_id,
            semestre=semestre,
            probabilidad=resultado["probabilidad"],
            prediccion=resultado["prediccion"],
            nivel_riesgo=resultado["nivel_riesgo"],
            umbral=resultado["umbral"],
            modelo_nombre=resultado["modelo"],
            modelo_version=resultado["version"],
            features=features,
            shap=shap_data,
        )
    finally:
        db.close()


def cargar_predicciones_db(engine, umbral_proba: float = 0.0) -> pd.DataFrame:
    from sqlalchemy.orm import sessionmaker
    from sapde.db.crud import obtener_predicciones_riesgo

    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        registros = obtener_predicciones_riesgo(db, umbral_proba, skip=0, limit=500)
        if not registros:
            return pd.DataFrame()
        rows = []
        for r in registros:
            rows.append({
                "student_id":   r.student_id,
                "semestre":     r.semestre,
                "probabilidad": round(r.probabilidad, 4),
                "nivel_riesgo": r.nivel_riesgo,
                "modelo":       r.modelo_nombre,
                "timestamp":    r.timestamp,
            })
        return pd.DataFrame(rows)
    finally:
        db.close()


def cargar_todos_artefactos() -> list[dict]:
    from sapde.models.artifacts import listar_artefactos
    return listar_artefactos(_MODELS_DIR)


def cargar_shap_resumen() -> dict[str, list[dict]]:
    """
    Carga shap_resumen.json y lo normaliza a:
      {modelo: [{"feature": str, "importancia_media": float}, ...]}
    Soporta tanto la estructura de lista como la de dict con
    keys 'top_features'/'importancias'.
    """
    import json
    if not _SHAP_REPORT.exists():
        return {}
    with open(_SHAP_REPORT, encoding="utf-8") as f:
        raw = json.load(f)

    normalizado: dict[str, list[dict]] = {}
    for modelo, val in raw.items():
        if isinstance(val, list):
            # Ya está en el formato correcto
            normalizado[modelo] = val
        elif isinstance(val, dict) and "importancias" in val:
            # Formato: {"top_features": [...], "importancias": {feat: val, ...}}
            normalizado[modelo] = [
                {"feature": feat, "importancia_media": imp}
                for feat, imp in val["importancias"].items()
            ]
        else:
            normalizado[modelo] = []
    return normalizado
