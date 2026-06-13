"""Endpoints de salud y métricas del sistema."""

from fastapi import APIRouter, Request
from sqlalchemy import text

from sapde.api.dependencies import DbDep
from sapde.api.schemas import ModeloInfo, RespuestaSalud

router = APIRouter(prefix="/health", tags=["health"])

_API_VERSION = "1.0.0"


@router.get("", response_model=RespuestaSalud, summary="Estado general de la API")
def health(request: Request, db: DbDep):
    modelo_nombre = None
    meta = getattr(request.app.state, "metadata", {})
    if meta:
        modelo_nombre = f"{meta.get('nombre')}_{meta.get('version')}"

    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return RespuestaSalud(
        estado="ok",
        version_api=_API_VERSION,
        modelo_activo=modelo_nombre,
        db_conectada=db_ok,
    )


@router.get(
    "/model",
    response_model=ModeloInfo,
    summary="Metadatos del modelo de predicción activo",
)
def model_info(request: Request):
    meta = getattr(request.app.state, "metadata", None)
    if not meta:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    return ModeloInfo(
        nombre=meta.get("nombre", ""),
        version=meta.get("version", ""),
        auc_roc=float(meta.get("auc_roc", 0.0)),
        recall=float(meta.get("recall", 0.0)),
        f1_score=float(meta.get("f1_score", 0.0)),
        umbral=float(meta.get("umbral", 0.5)),
        timestamp=str(meta.get("timestamp", "")),
    )
