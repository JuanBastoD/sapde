"""
Dependencias compartidas de FastAPI para SAPDE.

El modelo y el explainer se cargan una sola vez en el lifespan de la app
y se almacenan en app.state. Estas funciones los extraen para inyección.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from sapde.db.session import get_db


def get_modelo(request: Request):
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return modelo


def get_explainer(request: Request):
    explainer = getattr(request.app.state, "explainer", None)
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer SHAP no disponible")
    return explainer


def get_metadata(request: Request) -> dict:
    return getattr(request.app.state, "metadata", {})


DbDep = Annotated[Session, Depends(get_db)]
