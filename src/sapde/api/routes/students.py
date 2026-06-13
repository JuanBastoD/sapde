"""
Endpoints para consultar estudiantes en riesgo y sus explicaciones.

GET /students/at-risk            → listado paginado de estudiantes en riesgo
GET /students/{student_id}/explanation  → explicación detallada + recomendaciones
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Query, Request
from fastapi import HTTPException

from sapde.api.dependencies import DbDep, get_explainer, get_metadata
from sapde.api.schemas import (
    EstudianteRiesgo,
    ExplicacionSHAP,
    RespuestaAtRisk,
    RespuestaExplicacion,
)
from sapde.db import crud

logger = logging.getLogger("sapde.api.students")
router = APIRouter(prefix="/students", tags=["students"])


def _recomendaciones(nivel: str, top_riesgo: list[str]) -> list[str]:
    base = {
        "muy_alto": [
            "Iniciar proceso de acompañamiento académico urgente",
            "Citar al estudiante con consejero académico en los próximos 5 días hábiles",
            "Revisar posibilidad de matrícula diferencial el próximo semestre",
        ],
        "alto": [
            "Inscribir al estudiante en el programa de tutoría entre pares",
            "Monitorear avance académico mensualmente",
        ],
        "moderado": [
            "Alertar al monitor de grupo sobre el estudiante",
            "Enviar correo de seguimiento con recursos de apoyo",
        ],
        "bajo": [
            "Mantener seguimiento regular en el sistema",
        ],
    }
    recom = base.get(nivel, [])

    feature_recom = {
        "num_asignaturas_reprobadas": "Ofrecer refuerzo en las materias reprobadas",
        "promedio_acumulado": "Revisar plan de estudio y metodología personal",
        "interrupciones_previas": "Aplicar protocolo de reintegro para estudiantes con interrupciones",
        "tasa_reprobacion": "Gestionar nivelación académica con el departamento",
        "deficit_creditos": "Evaluar carga académica para el próximo semestre",
        "tendencia_calificaciones": "Citar para valorar factores externos que afectan el rendimiento",
    }
    for feat in top_riesgo:
        if feat in feature_recom and feature_recom[feat] not in recom:
            recom.append(feature_recom[feat])

    return recom


@router.get(
    "/at-risk",
    response_model=RespuestaAtRisk,
    summary="Listado de estudiantes en riesgo de deserción",
)
def at_risk(
    db: DbDep,
    umbral_proba: float = Query(0.5, ge=0.0, le=1.0, description="Umbral mínimo de probabilidad"),
    pagina: int = Query(1, ge=1),
    por_pagina: int = Query(20, ge=1, le=100),
):
    skip = (pagina - 1) * por_pagina
    total = crud.contar_predicciones_riesgo(db, umbral_proba)
    registros = crud.obtener_predicciones_riesgo(db, umbral_proba, skip, por_pagina)

    estudiantes = [
        EstudianteRiesgo(
            student_id=r.student_id,
            semestre=r.semestre,
            probabilidad_desercion=round(r.probabilidad, 4),
            nivel_riesgo=r.nivel_riesgo,
            ultima_prediccion=r.timestamp,
        )
        for r in registros
    ]

    return RespuestaAtRisk(
        total=total,
        pagina=pagina,
        por_pagina=por_pagina,
        estudiantes=estudiantes,
    )


@router.get(
    "/{student_id}/explanation",
    response_model=RespuestaExplicacion,
    summary="Explicación SHAP detallada para un estudiante",
)
def explanation(
    student_id: str,
    request: Request,
    db: DbDep,
):
    registro = crud.obtener_ultima_prediccion(db, student_id)
    if registro is None:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron predicciones para {student_id}",
        )

    shap_data = registro.shap_dict()
    if not shap_data:
        # Re-calcular SHAP en tiempo real
        features = registro.features_dict()
        if not features:
            raise HTTPException(
                status_code=422,
                detail="No hay features almacenadas para este estudiante",
            )
        try:
            import numpy as np

            explainer = get_explainer(request)
            feature_names: list[str] = request.app.state.feature_names
            x_raw = np.array([features[f] for f in feature_names], dtype=float)
            exp = explainer.explicar_estudiante(x_raw)

            sv = exp["por_feature"]
            sorted_sv = sorted(sv.items(), key=lambda kv: kv[1])
            top_riesgo = [k for k, v in sorted_sv if v > 0][-3:][::-1]
            top_protectores = [k for k, v in sorted_sv if v < 0][:3]

            shap_payload = ExplicacionSHAP(
                base_value=exp["base_value"],
                feature_contributions=sv,
                top_factores_riesgo=top_riesgo,
                top_factores_protectores=top_protectores,
            )
        except Exception as exc:
            logger.error("SHAP re-cálculo falló: %s", exc)
            raise HTTPException(status_code=503, detail="Explainer no disponible")
    else:
        shap_payload = ExplicacionSHAP(**shap_data)

    recom = _recomendaciones(registro.nivel_riesgo, shap_payload.top_factores_riesgo)

    return RespuestaExplicacion(
        student_id=student_id,
        probabilidad_desercion=round(registro.probabilidad, 4),
        nivel_riesgo=registro.nivel_riesgo,
        shap_explicacion=shap_payload,
        recomendaciones=recom,
        timestamp=registro.timestamp,
    )
