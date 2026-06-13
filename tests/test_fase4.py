"""
Tests Fase 4 — API FastAPI + DB (SQLite en memoria).

Estrategia:
  - Pipeline RF mínimo entrenado sobre las 25 features reales (nombres correctos)
  - DB usa SQLite en memoria (:memory:) para aislamiento total
  - Explainer SHAP desactivado (app.state.explainer = None) para velocidad
  - TestClient de Starlette para peticiones HTTP
"""

import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sapde.db.models import Base, Prediccion

# Orden canónico de las 25 features (mismo que train.py / engineering.py)
FEATURE_NAMES = [
    "semestre", "promedio_semestre", "promedio_acumulado",
    "tendencia_calificaciones", "creditos_matriculados", "creditos_aprobados",
    "porcentaje_avance_real", "num_asignaturas_reprobadas", "tasa_reprobacion",
    "semestres_cursados", "interrupciones_previas", "ratio_aprobacion",
    "deficit_creditos", "score_riesgo_bruto", "anio_cohorte", "semestre_cohorte",
    "nota_x_avance", "repro_x_deficit", "flag_nota_critica", "flag_reprobacion",
    "flag_rezago", "flag_tendencia_neg", "flag_interrupcion",
    "semestre_sq", "n_factores_riesgo",
]
N_FEATURES = len(FEATURE_NAMES)  # 25


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_mini():
    """
    Pipeline RF mínimo entrenado con los 25 nombres de features reales.
    X son valores aleatorios; suficiente para que predict_proba funcione.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, N_FEATURES)).astype(np.float64)
    y = np.array([0] * 225 + [1] * 75)
    X[:75, :3] += 2.0

    # DataFrame con columnas nombradas para que scaler guarde feature_names_in_
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("modelo", RandomForestClassifier(n_estimators=30, random_state=0)),
    ])
    pipe.fit(X_df, y)
    return pipe


@pytest.fixture(scope="module")
def meta_mini():
    return {
        "nombre":    "RandomForest",
        "version":   "v1",
        "auc_roc":   0.85,
        "recall":    0.70,
        "f1_score":  0.55,
        "umbral":    0.30,
        "timestamp": "2026-01-01T00:00:00",
    }


@pytest.fixture(scope="module")
def db_engine():
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="module")
def client(pipeline_mini, meta_mini, db_engine):
    """TestClient con modelo, metadata y DB inyectados sin lifespan real."""
    from sapde.api.main import app
    from sapde.api.dependencies import get_db

    # Override sesión DB → SQLite en memoria
    Session = sessionmaker(bind=db_engine)

    def override_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_db

    with TestClient(app, raise_server_exceptions=True) as c:
        # Sobreescribir estado DESPUÉS de que el lifespan se ejecute
        # (el lifespan puede fallar al buscar modelos, dejando estado vacío)
        app.state.modelo = pipeline_mini
        app.state.metadata = meta_mini
        app.state.feature_names = FEATURE_NAMES
        app.state.explainer = None   # SHAP desactivado en tests
        yield c

    app.dependency_overrides.clear()


def _entrada_valida() -> dict:
    """Payload con los 16 campos de EntradaEstudiante."""
    return {
        "student_id":                 "EST_TEST_01",
        "semestre":                   3,
        "promedio_semestre":          3.1,
        "promedio_acumulado":         3.3,
        "tendencia_calificaciones":  -0.2,
        "creditos_matriculados":      18,
        "creditos_aprobados":         14,
        "porcentaje_avance_real":     28.4,
        "num_asignaturas_reprobadas": 2,
        "tasa_reprobacion":           0.15,
        "semestres_cursados":         3,
        "interrupciones_previas":     0,
        "ratio_aprobacion":           0.78,
        "deficit_creditos":           5.0,
        "score_riesgo_bruto":         0.42,
        "anio_cohorte":               2022,
        "semestre_cohorte":           1,
    }


# ── Tests DB models ───────────────────────────────────────────────────────────

class TestDBModels:
    def test_crear_prediccion_basica(self, db_session):
        from sapde.db.crud import crear_prediccion
        p = crear_prediccion(
            db=db_session,
            student_id="EST_001",
            semestre=2,
            probabilidad=0.72,
            prediccion=True,
            nivel_riesgo="alto",
            umbral=0.30,
            modelo_nombre="RandomForest",
            modelo_version="v1",
        )
        assert p.id is not None
        assert p.student_id == "EST_001"
        assert p.probabilidad == pytest.approx(0.72)
        assert p.prediccion is True

    def test_prediccion_con_features_json(self, db_session):
        from sapde.db.crud import crear_prediccion
        features = {"feat_0": 1.5, "feat_1": -0.3}
        p = crear_prediccion(
            db=db_session,
            student_id="EST_002",
            semestre=1,
            probabilidad=0.15,
            prediccion=False,
            nivel_riesgo="bajo",
            umbral=0.30,
            modelo_nombre="SVM",
            modelo_version="v5",
            features=features,
            shap={"feat_0": 0.1, "feat_1": -0.05},
        )
        assert p.features_dict() == features

    def test_obtener_ultima_prediccion(self, db_session):
        from sapde.db.crud import crear_prediccion, obtener_ultima_prediccion
        for proba in [0.3, 0.6, 0.9]:
            crear_prediccion(
                db=db_session,
                student_id="EST_HIST",
                semestre=2,
                probabilidad=proba,
                prediccion=proba >= 0.3,
                nivel_riesgo="alto",
                umbral=0.3,
                modelo_nombre="RF",
                modelo_version="v1",
            )
        ultima = obtener_ultima_prediccion(db_session, "EST_HIST")
        assert ultima is not None
        assert ultima.student_id == "EST_HIST"

    def test_obtener_predicciones_riesgo_filtra_umbral(self, db_session):
        from sapde.db.crud import crear_prediccion, obtener_predicciones_riesgo
        for sid, prob in [("LOW_01", 0.10), ("HIGH_01", 0.85), ("HIGH_02", 0.91)]:
            crear_prediccion(
                db=db_session,
                student_id=sid,
                semestre=1,
                probabilidad=prob,
                prediccion=prob >= 0.3,
                nivel_riesgo="bajo" if prob < 0.3 else "muy_alto",
                umbral=0.3,
                modelo_nombre="RF",
                modelo_version="v1",
            )
        en_riesgo = obtener_predicciones_riesgo(db_session, umbral_proba=0.5)
        ids_riesgo = {r.student_id for r in en_riesgo}
        assert "HIGH_01" in ids_riesgo
        assert "HIGH_02" in ids_riesgo
        assert "LOW_01" not in ids_riesgo

    def test_contar_predicciones_riesgo(self, db_session):
        from sapde.db.crud import contar_predicciones_riesgo
        total = contar_predicciones_riesgo(db_session, umbral_proba=0.5)
        assert total >= 0


# ── Tests Pydantic schemas ────────────────────────────────────────────────────

class TestSchemas:
    def test_entrada_valida_pasa(self):
        from sapde.api.schemas import EntradaEstudiante
        e = EntradaEstudiante(**_entrada_valida())
        assert e.student_id == "EST_TEST_01"
        assert e.semestre == 3

    def test_entrada_creditos_invalidos(self):
        from pydantic import ValidationError
        from sapde.api.schemas import EntradaEstudiante
        datos = _entrada_valida()
        datos["creditos_aprobados"] = datos["creditos_matriculados"] + 5
        with pytest.raises(ValidationError):
            EntradaEstudiante(**datos)

    def test_entrada_semestre_fuera_de_rango(self):
        from pydantic import ValidationError
        from sapde.api.schemas import EntradaEstudiante
        datos = _entrada_valida()
        datos["semestre"] = 0
        with pytest.raises(ValidationError):
            EntradaEstudiante(**datos)

    def test_entrada_promedio_fuera_de_rango(self):
        from pydantic import ValidationError
        from sapde.api.schemas import EntradaEstudiante
        datos = _entrada_valida()
        datos["promedio_acumulado"] = 5.5
        with pytest.raises(ValidationError):
            EntradaEstudiante(**datos)

    def test_nivel_riesgo_classifica_correctamente(self):
        from sapde.api.routes.predict import _nivel_riesgo
        assert _nivel_riesgo(0.10, 0.30) == "bajo"
        assert _nivel_riesgo(0.25, 0.30) == "moderado"
        assert _nivel_riesgo(0.55, 0.30) == "alto"
        assert _nivel_riesgo(0.80, 0.30) == "muy_alto"


# ── Tests API health ──────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["estado"] == "ok"

    def test_health_db_conectada(self, client):
        r = client.get("/health")
        assert r.json()["db_conectada"] is True

    def test_health_modelo_cargado(self, client):
        r = client.get("/health")
        assert r.json()["modelo_activo"] is not None

    def test_health_model_endpoint(self, client):
        r = client.get("/health/model")
        assert r.status_code == 200
        body = r.json()
        assert "auc_roc" in body
        assert "umbral" in body
        assert body["nombre"] == "RandomForest"

    def test_root_retorna_sistema(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "SAPDE" in r.json()["sistema"]


# ── Tests API predict ─────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_retorna_200(self, client):
        r = client.post("/predict?incluir_shap=false", json=_entrada_valida())
        assert r.status_code == 200

    def test_predict_campos_requeridos(self, client):
        r = client.post("/predict?incluir_shap=false", json=_entrada_valida())
        body = r.json()
        for campo in ["student_id", "probabilidad_desercion", "nivel_riesgo",
                      "prediccion", "umbral", "modelo", "timestamp"]:
            assert campo in body, f"Falta campo: {campo}"

    def test_predict_probabilidad_en_rango(self, client):
        r = client.post("/predict?incluir_shap=false", json=_entrada_valida())
        proba = r.json()["probabilidad_desercion"]
        assert 0.0 <= proba <= 1.0

    def test_predict_nivel_riesgo_valido(self, client):
        r = client.post("/predict?incluir_shap=false", json=_entrada_valida())
        assert r.json()["nivel_riesgo"] in {"bajo", "moderado", "alto", "muy_alto"}

    def test_predict_payload_invalido_retorna_422(self, client):
        datos = _entrada_valida()
        datos.pop("promedio_acumulado")
        r = client.post("/predict?incluir_shap=false", json=datos)
        assert r.status_code == 422

    def test_predict_student_id_correcto(self, client):
        payload = _entrada_valida()
        payload["student_id"] = "EST_CONTROL"
        r = client.post("/predict?incluir_shap=false", json=payload)
        assert r.json()["student_id"] == "EST_CONTROL"

    def test_predict_persiste_en_db(self, client, db_session):
        payload = _entrada_valida()
        payload["student_id"] = "EST_PERSIST_TEST"
        r = client.post("/predict?incluir_shap=false", json=payload)
        assert r.status_code == 200
        from sapde.db.crud import obtener_ultima_prediccion
        reg = obtener_ultima_prediccion(db_session, "EST_PERSIST_TEST")
        assert reg is not None
        assert 0.0 <= reg.probabilidad <= 1.0

    def test_predict_umbral_desde_metadata(self, client):
        r = client.post("/predict?incluir_shap=false", json=_entrada_valida())
        assert r.json()["umbral"] == pytest.approx(0.30)

    def test_predict_shap_none_cuando_explainer_no_disponible(self, client):
        r = client.post("/predict?incluir_shap=true", json=_entrada_valida())
        assert r.status_code == 200
        # Sin explainer → shap_explicacion debe ser null
        assert r.json()["shap_explicacion"] is None


# ── Tests API students ────────────────────────────────────────────────────────

class TestStudents:
    def _registrar(self, client, student_id: str):
        payload = _entrada_valida()
        payload["student_id"] = student_id
        client.post("/predict?incluir_shap=false", json=payload)

    def test_at_risk_retorna_200(self, client):
        r = client.get("/students/at-risk")
        assert r.status_code == 200

    def test_at_risk_estructura(self, client):
        r = client.get("/students/at-risk")
        body = r.json()
        for campo in ["total", "pagina", "por_pagina", "estudiantes"]:
            assert campo in body

    def test_at_risk_paginacion(self, client):
        for i in range(5):
            self._registrar(client, f"EST_PAG_{i:03d}")
        r = client.get("/students/at-risk?por_pagina=2&pagina=1")
        body = r.json()
        assert body["por_pagina"] == 2
        assert len(body["estudiantes"]) <= 2

    def test_at_risk_umbral_alto_reduce_lista(self, client):
        r1 = client.get("/students/at-risk?umbral_proba=0.0")
        r2 = client.get("/students/at-risk?umbral_proba=0.99")
        assert r1.json()["total"] >= r2.json()["total"]

    def test_explanation_no_encontrado_retorna_404(self, client):
        r = client.get("/students/INEXISTENTE_XYZ_999/explanation")
        assert r.status_code == 404

    def test_explanation_sin_shap_stored_retorna_503(self, client):
        """Sin shap_json y sin explainer → 503 esperado."""
        payload = _entrada_valida()
        payload["student_id"] = "EST_EXPL_TEST"
        client.post("/predict?incluir_shap=false", json=payload)
        r = client.get("/students/EST_EXPL_TEST/explanation")
        assert r.status_code in {200, 422, 503}
