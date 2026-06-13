"""Operaciones CRUD sobre la tabla predicciones."""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from sapde.db.models import Prediccion


def crear_prediccion(
    db: Session,
    student_id: str,
    semestre: int,
    probabilidad: float,
    prediccion: bool,
    nivel_riesgo: str,
    umbral: float,
    modelo_nombre: str,
    modelo_version: str,
    features: Optional[dict] = None,
    shap: Optional[dict] = None,
) -> Prediccion:
    registro = Prediccion(
        student_id=student_id,
        semestre=semestre,
        probabilidad=probabilidad,
        prediccion=prediccion,
        nivel_riesgo=nivel_riesgo,
        umbral=umbral,
        modelo_nombre=modelo_nombre,
        modelo_version=modelo_version,
        features_json=json.dumps(features) if features else None,
        shap_json=json.dumps(shap) if shap else None,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)
    return registro


def obtener_predicciones_riesgo(
    db: Session,
    umbral_proba: float = 0.5,
    skip: int = 0,
    limit: int = 100,
) -> list[Prediccion]:
    """Devuelve las predicciones más recientes por encima del umbral dado."""
    from sqlalchemy import desc

    # Subquery: última predicción por estudiante
    from sqlalchemy import func
    subq = (
        db.query(
            Prediccion.student_id,
            func.max(Prediccion.timestamp).label("max_ts"),
        )
        .group_by(Prediccion.student_id)
        .subquery()
    )

    return (
        db.query(Prediccion)
        .join(
            subq,
            (Prediccion.student_id == subq.c.student_id)
            & (Prediccion.timestamp == subq.c.max_ts),
        )
        .filter(Prediccion.probabilidad >= umbral_proba)
        .order_by(desc(Prediccion.probabilidad))
        .offset(skip)
        .limit(limit)
        .all()
    )


def obtener_ultima_prediccion(
    db: Session, student_id: str
) -> Optional[Prediccion]:
    from sqlalchemy import desc

    return (
        db.query(Prediccion)
        .filter(Prediccion.student_id == student_id)
        .order_by(desc(Prediccion.timestamp))
        .first()
    )


def contar_predicciones_riesgo(db: Session, umbral_proba: float = 0.5) -> int:
    from sqlalchemy import func

    subq = (
        db.query(
            Prediccion.student_id,
            func.max(Prediccion.timestamp).label("max_ts"),
        )
        .group_by(Prediccion.student_id)
        .subquery()
    )

    return (
        db.query(Prediccion)
        .join(
            subq,
            (Prediccion.student_id == subq.c.student_id)
            & (Prediccion.timestamp == subq.c.max_ts),
        )
        .filter(Prediccion.probabilidad >= umbral_proba)
        .count()
    )
