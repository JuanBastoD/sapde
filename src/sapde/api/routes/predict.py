"""
POST /predict — predicción de deserción para un estudiante.

Flujo:
  1. Recibe 16 features crudas del ETL
  2. Construye DataFrame y agrega las 9 features avanzadas
  3. Ejecuta pipeline.predict_proba()
  4. Calcula explicación SHAP (opcional)
  5. Persiste en DB y devuelve RespuestaPrediccion
"""

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from sapde.api.dependencies import DbDep, get_explainer, get_metadata, get_modelo
from sapde.api.schemas import EntradaEstudiante, ExplicacionSHAP, RespuestaPrediccion
from sapde.db import crud

logger = logging.getLogger("sapde.api.predict")
router = APIRouter(prefix="/predict", tags=["predict"])


def _nivel_riesgo(proba: float, umbral: float) -> str:
    if proba < 0.20:
        return "bajo"
    if proba < umbral:
        return "moderado"
    if proba < 0.70:
        return "alto"
    return "muy_alto"


def _entrada_a_df(entrada: EntradaEstudiante) -> pd.DataFrame:
    """Convierte la entrada a DataFrame con las 16 features crudas + features avanzadas."""
    from sapde.features.engineering import agregar_features_avanzadas

    fila = {
        "semestre":                entrada.semestre,
        "promedio_semestre":       entrada.promedio_semestre,
        "promedio_acumulado":      entrada.promedio_acumulado,
        "tendencia_calificaciones": entrada.tendencia_calificaciones,
        "creditos_matriculados":   entrada.creditos_matriculados,
        "creditos_aprobados":      entrada.creditos_aprobados,
        "porcentaje_avance_real":  entrada.porcentaje_avance_real,
        "num_asignaturas_reprobadas": entrada.num_asignaturas_reprobadas,
        "tasa_reprobacion":        entrada.tasa_reprobacion,
        "semestres_cursados":      entrada.semestres_cursados,
        "interrupciones_previas":  entrada.interrupciones_previas,
        "ratio_aprobacion":        entrada.ratio_aprobacion,
        "deficit_creditos":        entrada.deficit_creditos,
        "score_riesgo_bruto":      entrada.score_riesgo_bruto,
        "anio_cohorte":            entrada.anio_cohorte,
        "semestre_cohorte":        entrada.semestre_cohorte,
        # columnas auxiliares que agregar_features_avanzadas puede requerir
        "student_id":              entrada.student_id,
        "deserta":                 0,
    }
    df = pd.DataFrame([fila])
    df = agregar_features_avanzadas(df)
    return df


def _features_array(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Extrae el array numpy en el orden exacto que espera el pipeline."""
    return df[feature_names].values.astype(float)


def _construir_shap(explicacion: dict) -> ExplicacionSHAP:
    sv = explicacion["por_feature"]
    sorted_sv = sorted(sv.items(), key=lambda kv: kv[1])
    top_riesgo = [k for k, v in sorted_sv if v > 0][-3:][::-1]
    top_protectores = [k for k, v in sorted_sv if v < 0][:3]

    return ExplicacionSHAP(
        base_value=explicacion["base_value"],
        feature_contributions=sv,
        top_factores_riesgo=top_riesgo,
        top_factores_protectores=top_protectores,
    )


@router.post("", response_model=RespuestaPrediccion, summary="Predice riesgo de deserción")
def predecir(
    entrada: EntradaEstudiante,
    request: Request,
    db: DbDep,
    incluir_shap: bool = Query(True, description="Incluir explicación SHAP en la respuesta"),
):
    pipeline = get_modelo(request)
    meta = get_metadata(request)
    feature_names: list[str] = request.app.state.feature_names

    umbral = float(meta.get("umbral", 0.5))

    df = _entrada_a_df(entrada)
    X = _features_array(df, feature_names)

    proba = float(pipeline.predict_proba(X)[0, 1])
    pred_bool = proba >= umbral
    nivel = _nivel_riesgo(proba, umbral)

    shap_payload: ExplicacionSHAP | None = None
    shap_dict: dict | None = None
    if incluir_shap:
        try:
            explainer = get_explainer(request)
            explicacion = explainer.explicar_estudiante(X[0])
            shap_payload = _construir_shap(explicacion)
            shap_dict = shap_payload.model_dump()
        except Exception as exc:
            logger.warning("SHAP falló: %s", exc)

    crud.crear_prediccion(
        db=db,
        student_id=entrada.student_id,
        semestre=entrada.semestre,
        probabilidad=proba,
        prediccion=pred_bool,
        nivel_riesgo=nivel,
        umbral=umbral,
        modelo_nombre=meta.get("nombre", ""),
        modelo_version=meta.get("version", ""),
        features=df[feature_names].iloc[0].to_dict(),
        shap=shap_dict,
    )

    return RespuestaPrediccion(
        student_id=entrada.student_id,
        semestre=entrada.semestre,
        probabilidad_desercion=round(proba, 4),
        nivel_riesgo=nivel,
        prediccion=pred_bool,
        umbral=umbral,
        modelo=meta.get("nombre", ""),
        version=meta.get("version", ""),
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        shap_explicacion=shap_payload,
    )
