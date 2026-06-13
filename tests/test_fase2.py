"""
Tests de la Fase 2: modelos ML, ANFIS, evaluación y artefactos.

Ejecutar:
    cd sapde
    pytest tests/test_fase2.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset_pequeno():
    """Dataset de prueba pequeño (150 estudiantes) para tests rápidos."""
    from sapde.data.synthetic import generar_dataset
    from sapde.etl.fallback import etl_python
    from sapde.features.engineering import preparar_dataset_ml, agregar_features_avanzadas

    df_raw = generar_dataset(n_estudiantes=150, semilla=42)
    df_etl = etl_python(df_raw, n_jobs=1)
    df_feat = agregar_features_avanzadas(df_etl)
    X, y = preparar_dataset_ml(df_feat)
    return X.values, y.values


@pytest.fixture(scope="module")
def split_datos(dataset_pequeno):
    """Split estratificado para tests de entrenamiento."""
    from sklearn.model_selection import train_test_split
    X, y = dataset_pequeno
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# ── Tests ANFIS ───────────────────────────────────────────────────────────────

class TestANFIS:

    def test_fit_predict(self, dataset_pequeno):
        """ANFIS entrena y predice sin errores."""
        from sapde.models.anfis import ANFISClassifier
        X, y = dataset_pequeno
        clf = ANFISClassifier(n_reglas=4, n_componentes_pca=6,
                              max_iter=50, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, dataset_pequeno):
        """ANFIS devuelve probabilidades válidas en [0, 1]."""
        from sapde.models.anfis import ANFISClassifier
        X, y = dataset_pequeno
        clf = ANFISClassifier(n_reglas=4, n_componentes_pca=6,
                              max_iter=50, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_sklearn_compatible(self, dataset_pequeno):
        """ANFIS es compatible con sklearn (clone, get_params, set_params)."""
        from sapde.models.anfis import ANFISClassifier
        from sklearn.base import clone
        X, y = dataset_pequeno
        clf = ANFISClassifier(n_reglas=4, n_componentes_pca=6,
                              max_iter=30, random_state=42)
        params = clf.get_params()
        assert "n_reglas" in params
        clf2 = clone(clf)
        clf2.fit(X, y)
        assert hasattr(clf2, "centros_")

    def test_reglas_activas(self, dataset_pequeno):
        """get_reglas_activas devuelve índice de regla por muestra."""
        from sapde.models.anfis import ANFISClassifier
        X, y = dataset_pequeno
        clf = ANFISClassifier(n_reglas=4, n_componentes_pca=6,
                              max_iter=30, random_state=42)
        clf.fit(X, y)
        reglas = clf.get_reglas_activas(X)
        assert reglas.shape == (len(X),)
        assert set(reglas).issubset(set(range(4)))

    def test_reproducibilidad_anfis(self, dataset_pequeno):
        """Mismo random_state produce el mismo resultado."""
        from sapde.models.anfis import ANFISClassifier
        X, y = dataset_pequeno
        clf1 = ANFISClassifier(n_reglas=3, max_iter=30, random_state=7)
        clf2 = ANFISClassifier(n_reglas=3, max_iter=30, random_state=7)
        clf1.fit(X, y); clf2.fit(X, y)
        np.testing.assert_allclose(clf1.centros_, clf2.centros_, rtol=1e-5)

    def test_auc_anfis_mejor_que_aleatorio(self, split_datos):
        """ANFIS alcanza AUC > 0.5 en test set."""
        from sapde.models.anfis import ANFISClassifier
        from sklearn.metrics import roc_auc_score
        X_tr, X_te, y_tr, y_te = split_datos
        clf = ANFISClassifier(n_reglas=4, n_componentes_pca=6,
                              max_iter=80, random_state=42)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        assert auc > 0.50, f"AUC ANFIS = {auc:.3f} <= 0.50"


# ── Tests Pipeline ML ─────────────────────────────────────────────────────────

class TestPipelineML:

    def test_crear_pipeline_tiene_pasos_correctos(self):
        """El pipeline tiene exactamente 3 pasos: smote, scaler, modelo."""
        from sapde.models.anfis import ANFISClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        clf = ANFISClassifier(n_reglas=3, max_iter=10)
        pipe = crear_pipeline(clf)
        assert "smote"  in dict(pipe.steps)
        assert "scaler" in dict(pipe.steps)
        assert "modelo" in dict(pipe.steps)

    def test_pipeline_rf_fit_predict(self, split_datos):
        """Pipeline con RF entrena y predice correctamente."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        X_tr, X_te, y_tr, y_te = split_datos
        rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)
        assert preds.shape == (len(X_te),)
        assert proba.shape == (len(X_te), 2)
        assert set(preds).issubset({0, 1})

    def test_obtener_modelos_retorna_cinco(self):
        """obtener_modelos() retorna al menos 5 modelos definidos."""
        from sapde.models.pipeline_ml import obtener_modelos
        modelos = obtener_modelos(random_state=42, n_jobs=1)
        assert len(modelos) >= 5
        assert "SVM" in modelos
        assert "MLP" in modelos
        assert "ANFIS" in modelos
        assert "RandomForest" in modelos

    def test_smote_solo_en_train(self, split_datos):
        """SMOTE no filtra datos de test (el tamaño de X_test no cambia)."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        X_tr, X_te, y_tr, y_te = split_datos
        rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)
        # Si SMOTE se aplicara en test, predict fallaría o el tamaño cambiaría
        preds = pipe.predict(X_te)
        assert len(preds) == len(X_te)


# ── Tests Evaluador ───────────────────────────────────────────────────────────

class TestEvaluador:

    def test_calibrar_umbral_youden_rango(self, split_datos):
        """El umbral Youden está en (0, 1)."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.evaluator import calibrar_umbral_youden
        X_tr, X_te, y_tr, y_te = split_datos
        rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)
        y_proba = pipe.predict_proba(X_tr)[:, 1]
        umbral = calibrar_umbral_youden(y_tr, y_proba)
        assert 0.0 < umbral < 1.0

    def test_evaluar_modelo_tiene_todas_metricas(self, split_datos):
        """evaluar_modelo() devuelve el dict completo de métricas OE2."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.evaluator import evaluar_modelo
        X_tr, X_te, y_tr, y_te = split_datos
        rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)
        metricas = evaluar_modelo("TestRF", pipe, X_te, y_te)
        for campo in ["precision", "recall", "f1_score", "auc_roc", "auc_pr",
                      "umbral", "TP", "TN", "FP", "FN"]:
            assert campo in metricas, f"Campo faltante: {campo}"

    def test_auc_roc_rango(self, split_datos):
        """AUC-ROC está en [0, 1]."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.evaluator import evaluar_modelo
        X_tr, X_te, y_tr, y_te = split_datos
        rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)
        m = evaluar_modelo("TestRF", pipe, X_te, y_te)
        assert 0.0 <= m["auc_roc"] <= 1.0

    def test_rf_auc_mejor_que_aleatorio(self, split_datos):
        """RandomForest alcanza AUC > 0.5."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.evaluator import evaluar_modelo
        X_tr, X_te, y_tr, y_te = split_datos
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)
        m = evaluar_modelo("RF", pipe, X_te, y_te)
        assert m["auc_roc"] > 0.50, f"AUC RF = {m['auc_roc']:.3f}"

    def test_tabla_comparacion_ordenada(self):
        """tabla_comparacion() ordena de mayor a menor AUC-ROC."""
        from sapde.models.evaluator import tabla_comparacion
        resultados = [
            {"modelo": "A", "auc_roc": 0.70, "precision": 0.7, "recall": 0.7,
             "f1_score": 0.7, "accuracy": 0.7, "auc_pr": 0.5, "umbral": 0.5,
             "t_entrenamiento_s": 1.0, "TP": 10, "TN": 80, "FP": 5, "FN": 5},
            {"modelo": "B", "auc_roc": 0.85, "precision": 0.8, "recall": 0.8,
             "f1_score": 0.8, "accuracy": 0.8, "auc_pr": 0.7, "umbral": 0.4,
             "t_entrenamiento_s": 2.0, "TP": 15, "TN": 80, "FP": 5, "FN": 5},
        ]
        tabla = tabla_comparacion(resultados)
        assert tabla.iloc[0]["modelo"] == "B"  # mayor AUC primero
        assert tabla.iloc[1]["modelo"] == "A"


# ── Tests Artefactos ──────────────────────────────────────────────────────────

class TestArtefactos:

    def test_guardar_y_cargar_pipeline(self, split_datos, tmp_path):
        """Pipeline se guarda y carga sin pérdida."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.artifacts import guardar_pipeline, cargar_pipeline
        X_tr, X_te, y_tr, y_te = split_datos

        rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)

        metricas = {"auc_roc": 0.72, "recall": 0.65, "f1_score": 0.61,
                    "precision": 0.58}
        ruta = guardar_pipeline("TestRF", pipe, metricas, 0.35,
                                directorio=tmp_path)
        assert ruta.exists()
        assert ruta.suffix == ".joblib"

        pipe2, meta = cargar_pipeline("TestRF", directorio=tmp_path)
        preds_orig  = pipe.predict(X_te)
        preds_carga = pipe2.predict(X_te)
        np.testing.assert_array_equal(preds_orig, preds_carga)
        assert meta["umbral"] == 0.35

    def test_listar_artefactos(self, split_datos, tmp_path):
        """listar_artefactos() enumera los modelos guardados."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.artifacts import guardar_pipeline, listar_artefactos
        X_tr, _, y_tr, _ = split_datos

        for nombre in ["ModeloA", "ModeloB"]:
            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            pipe = crear_pipeline(rf)
            pipe.fit(X_tr, y_tr)
            guardar_pipeline(nombre, pipe, {"auc_roc": 0.7, "recall": 0.6,
                                            "f1_score": 0.6, "precision": 0.6},
                             0.4, directorio=tmp_path)

        artefactos = listar_artefactos(tmp_path)
        assert len(artefactos) == 2
        nombres = {a["nombre"] for a in artefactos}
        assert nombres == {"ModeloA", "ModeloB"}

    def test_mejor_modelo(self, split_datos, tmp_path):
        """mejor_modelo() retorna el artefacto con mayor AUC."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.artifacts import guardar_pipeline, mejor_modelo
        X_tr, _, y_tr, _ = split_datos

        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)

        guardar_pipeline("Malo",  pipe, {"auc_roc": 0.60, "recall": 0.5,
                                         "f1_score": 0.5, "precision": 0.5},
                         0.5, directorio=tmp_path)
        guardar_pipeline("Bueno", pipe, {"auc_roc": 0.90, "recall": 0.8,
                                         "f1_score": 0.8, "precision": 0.8},
                         0.4, directorio=tmp_path)

        _, meta = mejor_modelo(tmp_path)
        assert meta["nombre"] == "Bueno"

    def test_version_incremental(self, split_datos, tmp_path):
        """Cada guardado incrementa la versión del artefacto."""
        from sklearn.ensemble import RandomForestClassifier
        from sapde.models.pipeline_ml import crear_pipeline
        from sapde.models.artifacts import guardar_pipeline, listar_artefactos
        X_tr, _, y_tr, _ = split_datos
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        pipe = crear_pipeline(rf)
        pipe.fit(X_tr, y_tr)

        for _ in range(3):
            guardar_pipeline("VersionTest", pipe,
                             {"auc_roc": 0.7, "recall": 0.6,
                              "f1_score": 0.6, "precision": 0.6},
                             0.5, directorio=tmp_path)

        artefactos = listar_artefactos(tmp_path)
        versiones = [a["version"] for a in artefactos if a["nombre"] == "VersionTest"]
        assert sorted(versiones) == [1, 2, 3]
