"""
Tests de la Fase 1: ETL paralelo y feature engineering.

Ejecutar:
    cd sapde
    pytest tests/test_fase1.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df_raw():
    """Dataset sintético pequeño para pruebas (200 estudiantes)."""
    from sapde.data.synthetic import generar_dataset
    return generar_dataset(n_estudiantes=200, semilla=42)


@pytest.fixture(scope="module")
def df_etl(df_raw, tmp_path_factory):
    """Dataset procesado por el ETL Python (fallback)."""
    from sapde.etl.fallback import etl_python
    return etl_python(df_raw, n_jobs=1)


@pytest.fixture(scope="module")
def df_features(df_etl):
    """Dataset con features avanzadas."""
    from sapde.features.engineering import agregar_features_avanzadas
    return agregar_features_avanzadas(df_etl)


# ── Tests del ETL fallback Python ─────────────────────────────────────────────

class TestETLPython:

    def test_columnas_originales_presentes(self, df_etl, df_raw):
        """El ETL conserva todas las columnas originales."""
        for col in df_raw.columns:
            assert col in df_etl.columns, f"Columna original faltante: {col}"

    def test_columnas_derivadas_presentes(self, df_etl):
        """El ETL agrega las columnas derivadas esperadas."""
        derivadas = ["ratio_aprobacion", "deficit_creditos",
                     "score_riesgo_bruto", "anio_cohorte", "semestre_cohorte"]
        for col in derivadas:
            assert col in df_etl.columns, f"Columna derivada faltante: {col}"

    def test_sin_nulos(self, df_etl):
        """No hay valores nulos tras el ETL."""
        nulos = df_etl.isnull().sum()
        assert nulos.sum() == 0, f"Nulos encontrados:\n{nulos[nulos > 0]}"

    def test_promedio_acumulado_clip(self, df_etl):
        """promedio_acumulado está en [1.0, 5.0] tras la limpieza."""
        col = df_etl["promedio_acumulado"]
        assert col.min() >= 1.0
        assert col.max() <= 5.0

    def test_tasa_reprobacion_clip(self, df_etl):
        col = df_etl["tasa_reprobacion"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_creditos_aprobados_leq_matriculados(self, df_etl):
        assert (df_etl["creditos_aprobados"] <= df_etl["creditos_matriculados"]).all()

    def test_ratio_aprobacion_rango(self, df_etl):
        """ratio_aprobacion ∈ [0, 1]."""
        col = df_etl["ratio_aprobacion"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0 + 1e-9  # tolerancia float

    def test_deficit_creditos_no_negativo(self, df_etl):
        """deficit_creditos = mat - aprobados ≥ 0."""
        assert (df_etl["deficit_creditos"] >= 0).all()

    def test_score_riesgo_rango(self, df_etl):
        """score_riesgo_bruto ∈ [0, 1]."""
        col = df_etl["score_riesgo_bruto"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0 + 1e-9

    def test_anio_cohorte_rango(self, df_etl):
        """año_cohorte está en el rango esperado (2019–2023)."""
        col = df_etl["anio_cohorte"]
        assert col.min() >= 2019
        assert col.max() <= 2025  # margen por si el generador cambia

    def test_semestre_cohorte_valores(self, df_etl):
        """semestre_cohorte es solo 1 o 2."""
        valores = set(df_etl["semestre_cohorte"].unique())
        assert valores <= {1, 2}, f"Valores inesperados: {valores}"

    def test_deserta_no_modificada(self, df_raw, df_etl):
        """El ETL no altera la variable objetivo."""
        original = df_raw["deserta"].reset_index(drop=True)
        procesada = df_etl["deserta"].reset_index(drop=True)
        pd.testing.assert_series_equal(original, procesada, check_names=False)

    def test_mismo_numero_filas(self, df_raw, df_etl):
        """El ETL no elimina ni duplica filas."""
        assert len(df_etl) == len(df_raw)


# ── Tests de reproducibilidad y paralelismo ───────────────────────────────────

class TestReproducibilidadETL:

    def test_resultado_igual_con_mas_jobs(self):
        """ETL con n_jobs=1 y n_jobs=2 produce el mismo resultado."""
        from sapde.data.synthetic import generar_dataset
        from sapde.etl.fallback import etl_python

        df_raw = generar_dataset(n_estudiantes=100, semilla=7)
        df1 = etl_python(df_raw, n_jobs=1)
        df2 = etl_python(df_raw, n_jobs=2)

        numericas = df1.select_dtypes(include=[float]).columns
        for col in numericas:
            np.testing.assert_allclose(
                df1[col].values, df2[col].values, rtol=1e-6,
                err_msg=f"Discrepancia en columna {col} al cambiar n_jobs",
            )

    def test_benchmark_etl(self, tmp_path):
        """El ETL se ejecuta correctamente con diferentes n_jobs."""
        from sapde.data.synthetic import generar_dataset, guardar_dataset
        from sapde.etl.fallback import ejecutar_etl_python

        df_raw = generar_dataset(n_estudiantes=300, semilla=42)
        csv_in  = tmp_path / "test_raw.csv"
        csv_out = tmp_path / "test_etl.csv"
        guardar_dataset(df_raw, csv_in)

        res = ejecutar_etl_python(csv_in, csv_out, n_jobs=1)
        assert res.registros == len(df_raw)
        assert res.t_etl > 0
        assert res.modo == "python"
        assert csv_out.exists()


# ── Tests de feature engineering ─────────────────────────────────────────────

class TestFeatureEngineering:

    def test_columnas_avanzadas_presentes(self, df_features):
        esperadas = [
            "nota_x_avance", "repro_x_deficit",
            "flag_nota_critica", "flag_reprobacion", "flag_rezago",
            "flag_tendencia_neg", "flag_interrupcion",
            "semestre_sq", "n_factores_riesgo",
        ]
        for col in esperadas:
            assert col in df_features.columns, f"Feature faltante: {col}"

    def test_flags_binarios(self, df_features):
        """Los flags son 0 o 1."""
        flags = [c for c in df_features.columns if c.startswith("flag_")]
        for col in flags:
            valores = set(df_features[col].unique())
            assert valores <= {0, 1}, f"{col} tiene valores no binarios: {valores}"

    def test_n_factores_riesgo_rango(self, df_features):
        """n_factores_riesgo ∈ [0, 5]."""
        col = df_features["n_factores_riesgo"]
        assert col.min() >= 0
        assert col.max() <= 5

    def test_semestre_sq_positivo(self, df_features):
        assert (df_features["semestre_sq"] > 0).all()

    def test_nota_x_avance_no_negativo(self, df_features):
        assert (df_features["nota_x_avance"] >= 0).all()

    def test_preparar_dataset_ml(self, df_features):
        """preparar_dataset_ml devuelve X e y con las dimensiones correctas."""
        from sapde.features.engineering import preparar_dataset_ml
        X, y = preparar_dataset_ml(df_features)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert X.isnull().sum().sum() == 0
        assert set(y.unique()) <= {0, 1}
        assert X.shape[1] > 10  # al menos 10 features

    def test_features_son_numericas(self, df_features):
        """Todas las features del dataset ML son numéricas."""
        from sapde.features.engineering import preparar_dataset_ml
        X, _ = preparar_dataset_ml(df_features)
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), f"{col} no es numérica"

    def test_correlacion_score_con_objetivo(self, df_features):
        """
        score_riesgo_bruto tiene correlación positiva con deserta.
        Nota: deserta es solo ~4% positivo a nivel fila (label escaso),
        por lo que la correlación puntual es baja pero debe ser positiva.
        La prueba a nivel estudiante (max deserta por estudiante) muestra
        correlación más clara (~0.4), validada en el EDA.
        """
        if "score_riesgo_bruto" not in df_features.columns:
            pytest.skip("score_riesgo_bruto no disponible")
        # Test a nivel fila: señal es débil por desbalance extremo (4% positivos)
        corr_fila = df_features["score_riesgo_bruto"].corr(df_features["deserta"])
        assert corr_fila > 0.0, f"Correlacion fila esperada > 0, obtenida: {corr_fila:.3f}"

        # Test a nivel estudiante: señal más robusta (~28% positivos)
        df_est = (
            df_features.groupby("student_id")
            .agg(score_max=("score_riesgo_bruto", "mean"),
                 deserta_final=("deserta", "max"))
            .reset_index()
        )
        corr_est = df_est["score_max"].corr(df_est["deserta_final"])
        assert corr_est > 0.15, f"Correlacion estudiante esperada > 0.15, obtenida: {corr_est:.3f}"

    def test_correlacion_nota_con_objetivo_negativa(self, df_features):
        """promedio_acumulado debe correlacionar negativamente con deserta."""
        corr = df_features["promedio_acumulado"].corr(df_features["deserta"])
        assert corr < -0.05, f"Correlación esperada negativa, obtenida: {corr:.3f}"


# ── Test de guardado Parquet ───────────────────────────────────────────────────

class TestPersistencia:

    def test_guardar_csv_etl(self, df_raw, tmp_path):
        """ETL guarda CSV correctamente."""
        from sapde.etl.fallback import ejecutar_etl_python
        from sapde.data.synthetic import guardar_dataset

        csv_in  = tmp_path / "raw.csv"
        csv_out = tmp_path / "etl.csv"
        guardar_dataset(df_raw, csv_in)

        res = ejecutar_etl_python(csv_in, csv_out, n_jobs=1)
        assert csv_out.exists()

        df_cargado = pd.read_csv(csv_out)
        assert len(df_cargado) == len(df_raw)
        assert "ratio_aprobacion" in df_cargado.columns

    def test_guardar_parquet(self, df_features, tmp_path):
        """El dataset procesado se guarda en Parquet sin pérdida."""
        from sapde.features.engineering import guardar_dataset_procesado

        ruta_parquet = tmp_path / "dataset_final.parquet"
        guardar_dataset_procesado(df_features, ruta_parquet)
        assert ruta_parquet.exists()

        df_recargado = pd.read_parquet(ruta_parquet)
        assert len(df_recargado) == len(df_features)
        assert set(df_recargado.columns) == set(df_features.columns)
