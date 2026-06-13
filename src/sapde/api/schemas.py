"""
Esquemas Pydantic para la API SAPDE (Fase 4).

EntradaEstudiante  → los 16 features crudos del ETL
                     (las 9 features avanzadas se derivan internamente)
RespuestaPrediccion → resultado de /predict
RespuestaExplicacion → resultado de /students/{id}/explanation
EstudianteRiesgo    → ítem del listado /students/at-risk
RespuestaAtRisk     → envoltorio paginado de lo anterior
ModeloInfo          → metadatos del modelo activo (/health/model)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Entrada ───────────────────────────────────────────────────────────────────

class EntradaEstudiante(BaseModel):
    """Features crudas de un estudiante para un semestre dado."""

    student_id: str = Field(..., description="ID único del estudiante")

    # Académicas básicas
    semestre: int = Field(..., ge=1, le=12, description="Semestre académico actual")
    promedio_semestre: float = Field(..., ge=0.0, le=5.0)
    promedio_acumulado: float = Field(..., ge=0.0, le=5.0)
    tendencia_calificaciones: float = Field(
        ..., description="Diferencia promedio_semestre - promedio_acumulado"
    )
    creditos_matriculados: int = Field(..., ge=0)
    creditos_aprobados: int = Field(..., ge=0)
    porcentaje_avance_real: float = Field(..., ge=0.0, le=100.0)
    num_asignaturas_reprobadas: int = Field(..., ge=0)
    tasa_reprobacion: float = Field(..., ge=0.0, le=1.0)

    # Historial
    semestres_cursados: int = Field(..., ge=0)
    interrupciones_previas: int = Field(..., ge=0)

    # Derivadas ETL
    ratio_aprobacion: float = Field(..., ge=0.0, le=1.0)
    deficit_creditos: float = Field(..., description="Créditos faltantes respecto al avance esperado")
    score_riesgo_bruto: float = Field(..., ge=0.0, le=1.0)

    # Cohorte desglosada
    anio_cohorte: int = Field(..., ge=2000, le=2030)
    semestre_cohorte: int = Field(..., ge=1, le=2)

    @field_validator("creditos_aprobados")
    @classmethod
    def aprobados_lte_matriculados(cls, v, info):
        data = info.data
        if "creditos_matriculados" in data and v > data["creditos_matriculados"]:
            raise ValueError(
                "creditos_aprobados no puede exceder creditos_matriculados"
            )
        return v


# ── Explicación SHAP ─────────────────────────────────────────────────────────

class ExplicacionSHAP(BaseModel):
    base_value: float
    feature_contributions: dict[str, float]
    top_factores_riesgo: list[str]
    top_factores_protectores: list[str]


# ── Respuesta de predicción ───────────────────────────────────────────────────

class RespuestaPrediccion(BaseModel):
    student_id: str
    semestre: int
    probabilidad_desercion: float
    nivel_riesgo: str   # bajo | moderado | alto | muy_alto
    prediccion: bool
    umbral: float
    modelo: str
    version: str
    timestamp: datetime
    shap_explicacion: Optional[ExplicacionSHAP] = None


# ── Detalle de explicación ────────────────────────────────────────────────────

class RespuestaExplicacion(BaseModel):
    student_id: str
    probabilidad_desercion: float
    nivel_riesgo: str
    shap_explicacion: ExplicacionSHAP
    recomendaciones: list[str]
    timestamp: datetime


# ── Listado at-risk ───────────────────────────────────────────────────────────

class EstudianteRiesgo(BaseModel):
    student_id: str
    semestre: int
    probabilidad_desercion: float
    nivel_riesgo: str
    ultima_prediccion: datetime


class RespuestaAtRisk(BaseModel):
    total: int
    pagina: int
    por_pagina: int
    estudiantes: list[EstudianteRiesgo]


# ── Información del modelo ────────────────────────────────────────────────────

class ModeloInfo(BaseModel):
    nombre: str
    version: str
    auc_roc: float
    recall: float
    f1_score: float
    umbral: float
    timestamp: str


# ── Respuesta de salud ────────────────────────────────────────────────────────

class RespuestaSalud(BaseModel):
    estado: str
    version_api: str
    modelo_activo: Optional[str]
    db_conectada: bool
