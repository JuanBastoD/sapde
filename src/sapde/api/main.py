"""
Aplicación FastAPI de SAPDE — Sistema de Analítica Predictiva de Deserción.

Lifespan:
  1. Carga el mejor modelo entrenado (mayor AUC-ROC)
  2. Prepara el explainer SHAP con background del dataset sintético
  3. Inicializa las tablas de la DB (PostgreSQL o SQLite)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("sapde.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

_RAIZ = Path(__file__).resolve().parents[3]


def _cargar_modelo():
    from sapde.models.artifacts import mejor_modelo
    pipeline, raw_meta = mejor_modelo(_RAIZ / "models")

    meta = {
        "nombre":    raw_meta.get("nombre", ""),
        "version":   f"v{raw_meta.get('version', '?')}",
        "auc_roc":   raw_meta.get("metricas", {}).get("auc_roc", 0.0),
        "recall":    raw_meta.get("metricas", {}).get("recall", 0.0),
        "f1_score":  raw_meta.get("metricas", {}).get("f1_score", 0.0),
        "umbral":    raw_meta.get("umbral", 0.5),
        "timestamp": raw_meta.get("timestamp", ""),
    }

    scaler = pipeline.named_steps["scaler"]
    if hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    else:
        # Orden canónico de las 25 features (mismo que train.py)
        feature_names = [
            "semestre", "promedio_semestre", "promedio_acumulado",
            "tendencia_calificaciones", "creditos_matriculados", "creditos_aprobados",
            "porcentaje_avance_real", "num_asignaturas_reprobadas", "tasa_reprobacion",
            "semestres_cursados", "interrupciones_previas", "ratio_aprobacion",
            "deficit_creditos", "score_riesgo_bruto", "anio_cohorte", "semestre_cohorte",
            "nota_x_avance", "repro_x_deficit", "flag_nota_critica", "flag_reprobacion",
            "flag_rezago", "flag_tendencia_neg", "flag_interrupcion",
            "semestre_sq", "n_factores_riesgo",
        ]

    return pipeline, meta, feature_names


def _cargar_background(feature_names: list[str]):
    """Carga 300 muestras del dataset sintético como background para SHAP."""
    import numpy as np
    import pandas as pd

    from sapde.etl.procesador import procesar_datos
    from sapde.features.engineering import preparar_dataset_ml

    csv = _RAIZ / "data" / "raw" / "estudiantes_sintetico.csv"
    df_raw = pd.read_csv(csv)

    df_etl = procesar_datos(df_raw)
    X, _ = preparar_dataset_ml(df_etl, incluir_features_avanzadas=True)

    # Asegura que las columnas estén en el orden correcto
    X = X[feature_names]

    rng = np.random.default_rng(42)
    n = min(300, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    return X.values[idx]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Modelo ─────────────────────────────────────────────────────────────
    try:
        pipeline, meta, feature_names = _cargar_modelo()
        app.state.modelo = pipeline
        app.state.metadata = meta
        app.state.feature_names = feature_names
        logger.info(
            "Modelo cargado: %s %s  AUC=%.3f  umbral=%.3f",
            meta["nombre"], meta["version"], meta["auc_roc"], meta["umbral"],
        )
    except Exception as exc:
        logger.error("No se pudo cargar el modelo: %s", exc)
        app.state.modelo = None
        app.state.metadata = {}
        app.state.feature_names = []

    # ── 2. Explainer SHAP ────────────────────────────────────────────────────
    app.state.explainer = None
    if app.state.modelo is not None:
        try:
            from sapde.explain.shap_explainer import ExplicadorSHAP

            X_bg = _cargar_background(app.state.feature_names)
            app.state.explainer = ExplicadorSHAP(
                nombre=app.state.metadata["nombre"],
                pipeline=app.state.modelo,
                feature_names=app.state.feature_names,
                X_background=X_bg,
                n_background=200,
            )
            logger.info("ExplicadorSHAP listo (%d muestras de fondo)", len(X_bg))
        except Exception as exc:
            logger.warning("SHAP no disponible: %s", exc)

    # ── 3. Base de datos ─────────────────────────────────────────────────────
    try:
        from sapde.db.session import init_db
        init_db()
    except Exception as exc:
        logger.error("Error inicializando DB: %s", exc)

    yield

    logger.info("API cerrando.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SAPDE — API de Predicción de Deserción Estudiantil",
    description=(
        "Sistema de Analítica Predictiva de Deserción Estudiantil. "
        "Universidad de Pamplona — Ingeniería de Sistemas."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rutas ─────────────────────────────────────────────────────────────────────
from sapde.api.routes import health, predict, students  # noqa: E402

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(students.router)


@app.get("/", tags=["root"])
def root():
    return {
        "sistema": "SAPDE",
        "descripcion": "API de predicción de deserción estudiantil",
        "docs": "/docs",
    }
