"""
Tests de la Fase 0: configuración y generación de datos sintéticos.

Ejecutar:
    cd sapde
    pytest tests/test_fase0.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ===========================================================
# Fixtures
# ===========================================================

@pytest.fixture(scope="module")
def dataset_sintetico():
    """Genera un dataset de prueba pequeño (200 estudiantes)."""
    from sapde.data.synthetic import generar_dataset
    return generar_dataset(n_estudiantes=200, semilla=42)


@pytest.fixture(scope="module")
def config():
    """Carga la configuración."""
    from sapde.config import obtener_config
    return obtener_config()


# ===========================================================
# Tests de configuración
# ===========================================================

class TestConfiguracion:
    def test_config_carga(self, config):
        """La configuración se carga sin excepciones."""
        assert config is not None

    def test_random_seed_valido(self, config):
        assert isinstance(config.random_seed, int)
        assert config.random_seed >= 0

    def test_n_estudiantes_positivo(self, config):
        assert config.synthetic_n_estudiantes > 0

    def test_database_url_formato(self, config):
        url = config.database_url
        assert url.startswith("postgresql://")
        assert config.postgres_host in url
        assert config.postgres_db in url

    def test_log_level_valido(self, config):
        niveles_validos = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert config.log_level in niveles_validos

    def test_test_size_rango(self, config):
        assert 0.05 < config.test_size < 0.50

    def test_singleton(self):
        """obtener_config() retorna siempre la misma instancia."""
        from sapde.config import obtener_config
        c1 = obtener_config()
        c2 = obtener_config()
        assert c1 is c2


# ===========================================================
# Tests del generador de datos
# ===========================================================

class TestDatasetSintetico:

    COLUMNAS_REQUERIDAS = {
        "student_id", "cohorte", "semestre",
        "promedio_semestre", "promedio_acumulado", "tendencia_calificaciones",
        "creditos_matriculados", "creditos_aprobados", "porcentaje_avance_real",
        "num_asignaturas_reprobadas", "tasa_reprobacion",
        "semestres_cursados", "interrupciones_previas",
        "deserta",
    }

    def test_columnas_correctas(self, dataset_sintetico):
        """El dataset tiene exactamente las columnas del esquema SAPDE."""
        assert set(dataset_sintetico.columns) == self.COLUMNAS_REQUERIDAS

    def test_sin_nulos(self, dataset_sintetico):
        """No hay valores nulos en el dataset."""
        assert dataset_sintetico.isnull().sum().sum() == 0

    def test_tamano_razonable(self, dataset_sintetico):
        """El dataset tiene al menos 200 filas (>=1 por estudiante)."""
        assert len(dataset_sintetico) >= 200

    def test_student_id_unicos_plausibles(self, dataset_sintetico):
        """Hay exactamente 200 estudiantes únicos."""
        n_unicos = dataset_sintetico["student_id"].nunique()
        assert n_unicos == 200

    def test_promedio_semestre_rango(self, dataset_sintetico):
        """Las notas están en el rango [1.0, 5.0]."""
        col = dataset_sintetico["promedio_semestre"]
        assert col.min() >= 1.0
        assert col.max() <= 5.0

    def test_promedio_acumulado_rango(self, dataset_sintetico):
        col = dataset_sintetico["promedio_acumulado"]
        assert col.min() >= 1.0
        assert col.max() <= 5.0

    def test_creditos_aprobados_leq_matriculados(self, dataset_sintetico):
        """Los créditos aprobados nunca superan los matriculados."""
        diferencia = (
            dataset_sintetico["creditos_aprobados"]
            > dataset_sintetico["creditos_matriculados"]
        )
        assert not diferencia.any(), "Hay filas con aprobados > matriculados"

    def test_creditos_no_negativos(self, dataset_sintetico):
        for col in ["creditos_matriculados", "creditos_aprobados"]:
            assert (dataset_sintetico[col] >= 0).all(), f"{col} tiene valores negativos"

    def test_porcentaje_avance_rango(self, dataset_sintetico):
        col = dataset_sintetico["porcentaje_avance_real"]
        assert col.min() >= 0.0
        assert col.max() <= 100.0

    def test_tasa_reprobacion_rango(self, dataset_sintetico):
        col = dataset_sintetico["tasa_reprobacion"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_deserta_binaria(self, dataset_sintetico):
        """La variable objetivo es binaria (0 o 1)."""
        valores = set(dataset_sintetico["deserta"].unique())
        assert valores <= {0, 1}, f"Valores no binarios: {valores}"

    def test_tasa_desercion_rango(self, dataset_sintetico):
        """La tasa de deserción está entre 15% y 40% (nivel estudiante)."""
        tasa = (
            dataset_sintetico.groupby("student_id")["deserta"].max().mean()
        )
        assert 0.15 <= tasa <= 0.40, f"Tasa de deserción inesperada: {tasa:.2%}"

    def test_reproducibilidad(self):
        """El mismo generador con la misma semilla produce el mismo resultado."""
        from sapde.data.synthetic import generar_dataset
        df1 = generar_dataset(n_estudiantes=100, semilla=99)
        df2 = generar_dataset(n_estudiantes=100, semilla=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_semilla_diferente_resultado_diferente(self):
        """Semillas distintas producen resultados distintos."""
        from sapde.data.synthetic import generar_dataset
        df1 = generar_dataset(n_estudiantes=100, semilla=1)
        df2 = generar_dataset(n_estudiantes=100, semilla=2)
        assert not df1["promedio_semestre"].equals(df2["promedio_semestre"])

    def test_semestres_cursados_coherente(self, dataset_sintetico):
        """semestres_cursados coincide con el número de semestre en la fila."""
        mismatch = dataset_sintetico["semestres_cursados"] != dataset_sintetico["semestre"]
        assert not mismatch.any(), "semestres_cursados no coincide con semestre"

    def test_interrupciones_no_negativas(self, dataset_sintetico):
        assert (dataset_sintetico["interrupciones_previas"] >= 0).all()

    def test_cohorte_formato(self, dataset_sintetico):
        """Las cohortes tienen el formato YYYY-I o YYYY-II."""
        import re
        patron = re.compile(r"^\d{4}-(I|II)$")
        invalidas = dataset_sintetico["cohorte"].apply(
            lambda c: not bool(patron.match(c))
        )
        assert not invalidas.any(), "Hay cohortes con formato incorrecto"

    def test_guardar_csv(self, dataset_sintetico, tmp_path):
        """El dataset se guarda y recarga sin pérdida de datos."""
        from sapde.data.synthetic import guardar_dataset
        ruta = tmp_path / "test_dataset.csv"
        guardar_dataset(dataset_sintetico, ruta)
        df_cargado = pd.read_csv(ruta)
        assert len(df_cargado) == len(dataset_sintetico)
        assert set(df_cargado.columns) == set(dataset_sintetico.columns)
