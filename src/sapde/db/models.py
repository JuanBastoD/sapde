"""
Modelos SQLAlchemy para SAPDE (Fase 4).

Tablas:
  - predicciones: historial de predicciones por estudiante-semestre
"""

import json
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediccion(Base):
    __tablename__ = "predicciones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    semestre: Mapped[int] = mapped_column(Integer, nullable=False)

    probabilidad: Mapped[float] = mapped_column(Float, nullable=False)
    prediccion: Mapped[bool] = mapped_column(Boolean, nullable=False)
    nivel_riesgo: Mapped[str] = mapped_column(String(20), nullable=False)
    umbral: Mapped[float] = mapped_column(Float, nullable=False)

    modelo_nombre: Mapped[str] = mapped_column(String(50), nullable=False)
    modelo_version: Mapped[str] = mapped_column(String(20), nullable=False)

    # Features y explicación guardadas como JSON
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    shap_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        nullable=False,
    )

    def features_dict(self) -> dict:
        return json.loads(self.features_json) if self.features_json else {}

    def shap_dict(self) -> dict:
        return json.loads(self.shap_json) if self.shap_json else {}
