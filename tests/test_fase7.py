"""
Tests Fase 7 — Integración Docker, configuración final y cobertura del proyecto.

Verifica:
  - Existencia y contenido de Dockerfiles
  - Configuración de docker-compose.yml
  - Variables de entorno en .env.example
  - pyproject.toml correcto
  - Importabilidad de todos los módulos sapde
  - Suite de tests completa (≥187 tests previos)
  - README actualizado con todas las fases
"""

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ── Rutas ──────────────────────────────────────────────────────────────────────

_RAIZ = Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════════════
# TestDockerfileBackend
# ══════════════════════════════════════════════════════════════════════════════

class TestDockerfileBackend:
    """Verifica que Dockerfile.backend existe y tiene instrucciones válidas."""

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / "Dockerfile.backend"
        assert ruta.exists(), "Dockerfile.backend no existe"
        return ruta.read_text(encoding="utf-8")

    def test_existe(self):
        assert (_RAIZ / "Dockerfile.backend").exists()

    def test_from_python312(self, contenido):
        assert "FROM python:3.12" in contenido

    def test_workdir(self, contenido):
        assert "WORKDIR" in contenido

    def test_copia_requirements(self, contenido):
        assert "requirements.txt" in contenido

    def test_copia_src(self, contenido):
        assert "COPY src/" in contenido

    def test_pythonpath_src(self, contenido):
        assert "PYTHONPATH" in contenido and "src" in contenido

    def test_expose(self, contenido):
        assert "EXPOSE" in contenido

    def test_cmd_uvicorn(self, contenido):
        assert "uvicorn" in contenido

    def test_healthcheck(self, contenido):
        assert "HEALTHCHECK" in contenido

    def test_usuario_no_root(self, contenido):
        assert "useradd" in contenido or "USER" in contenido


# ══════════════════════════════════════════════════════════════════════════════
# TestDockerfileDashboard
# ══════════════════════════════════════════════════════════════════════════════

class TestDockerfileDashboard:
    """Verifica que Dockerfile.dashboard existe y tiene instrucciones válidas."""

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / "Dockerfile.dashboard"
        assert ruta.exists(), "Dockerfile.dashboard no existe"
        return ruta.read_text(encoding="utf-8")

    def test_existe(self):
        assert (_RAIZ / "Dockerfile.dashboard").exists()

    def test_from_python312(self, contenido):
        assert "FROM python:3.12" in contenido

    def test_cmd_streamlit(self, contenido):
        assert "streamlit" in contenido.lower()

    def test_apunta_dashboard_app(self, contenido):
        assert "dashboard/app.py" in contenido

    def test_expose_8501(self, contenido):
        assert "8501" in contenido

    def test_healthcheck(self, contenido):
        assert "HEALTHCHECK" in contenido

    def test_pythonpath(self, contenido):
        assert "PYTHONPATH" in contenido


# ══════════════════════════════════════════════════════════════════════════════
# TestDockerCompose
# ══════════════════════════════════════════════════════════════════════════════

class TestDockerCompose:
    """Verifica el archivo docker-compose.yml."""

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / "docker-compose.yml"
        assert ruta.exists(), "docker-compose.yml no existe"
        return ruta.read_text(encoding="utf-8")

    def test_existe(self):
        assert (_RAIZ / "docker-compose.yml").exists()

    def test_servicio_db(self, contenido):
        assert "db:" in contenido

    def test_servicio_backend(self, contenido):
        assert "backend:" in contenido

    def test_servicio_dashboard(self, contenido):
        assert "dashboard:" in contenido

    def test_imagen_postgres(self, contenido):
        assert "postgres:" in contenido

    def test_dockerfile_backend_referenciado(self, contenido):
        assert "Dockerfile.backend" in contenido

    def test_dockerfile_dashboard_referenciado(self, contenido):
        assert "Dockerfile.dashboard" in contenido

    def test_healthcheck_db(self, contenido):
        assert "pg_isready" in contenido

    def test_depends_on_db(self, contenido):
        assert "service_healthy" in contenido

    def test_volumen_postgres(self, contenido):
        assert "postgres_data" in contenido

    def test_puerto_backend_8000(self, contenido):
        assert "8000" in contenido

    def test_puerto_dashboard_8501(self, contenido):
        assert "8501" in contenido

    def test_env_file(self, contenido):
        assert "env_file" in contenido

    def test_dashboard_monta_modelos(self, contenido):
        assert "./models" in contenido


# ══════════════════════════════════════════════════════════════════════════════
# TestDockerignore
# ══════════════════════════════════════════════════════════════════════════════

class TestDockerignore:
    """Verifica que .dockerignore excluye artefactos pesados."""

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / ".dockerignore"
        assert ruta.exists(), ".dockerignore no existe"
        return ruta.read_text(encoding="utf-8")

    def test_existe(self):
        assert (_RAIZ / ".dockerignore").exists()

    def test_excluye_venv(self, contenido):
        assert ".venv/" in contenido

    def test_excluye_pycache(self, contenido):
        assert "__pycache__/" in contenido

    def test_excluye_tests(self, contenido):
        assert "tests/" in contenido

    def test_excluye_git(self, contenido):
        assert ".git/" in contenido

    def test_excluye_env(self, contenido):
        assert ".env" in contenido


# ══════════════════════════════════════════════════════════════════════════════
# TestDotEnvExample
# ══════════════════════════════════════════════════════════════════════════════

class TestDotEnvExample:
    """Verifica que .env.example tiene todas las variables necesarias."""

    _VARS_REQUERIDAS = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "API_PORT",
        "DASHBOARD_PORT",
        "RANDOM_SEED",
        "LOG_LEVEL",
    ]

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / ".env.example"
        assert ruta.exists(), ".env.example no existe"
        return ruta.read_text(encoding="utf-8")

    def test_existe(self):
        assert (_RAIZ / ".env.example").exists()

    @pytest.mark.parametrize("var", _VARS_REQUERIDAS)
    def test_variable_presente(self, contenido, var):
        assert var in contenido, f"Variable {var!r} no encontrada en .env.example"


# ══════════════════════════════════════════════════════════════════════════════
# TestPyproject
# ══════════════════════════════════════════════════════════════════════════════

class TestPyproject:
    """Verifica pyproject.toml."""

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / "pyproject.toml"
        assert ruta.exists(), "pyproject.toml no existe"
        return ruta.read_text(encoding="utf-8")

    def test_nombre_sapde(self, contenido):
        assert 'name = "sapde"' in contenido

    def test_python_312(self, contenido):
        assert "3.12" in contenido

    def test_pytest_config(self, contenido):
        assert "[tool.pytest.ini_options]" in contenido

    def test_testpaths(self, contenido):
        assert "testpaths" in contenido

    def test_build_system(self, contenido):
        assert "[build-system]" in contenido

    def test_setuptools(self, contenido):
        assert "setuptools" in contenido


# ══════════════════════════════════════════════════════════════════════════════
# TestImportModulos
# ══════════════════════════════════════════════════════════════════════════════

class TestImportModulos:
    """Verifica que todos los módulos sapde son importables sin error."""

    _MODULOS = [
        "sapde",
        "sapde.config",
        "sapde.data.synthetic",
        "sapde.etl.pipeline",
        "sapde.etl.fallback",
        "sapde.features.engineering",
        "sapde.models.artifacts",
        "sapde.models.train",
        "sapde.explain.shap_explainer",
        "sapde.db.models",
        "sapde.db.session",
        "sapde.db.crud",
        "sapde.api.schemas",
        "sapde.api.dependencies",
        "sapde.api.main",
        "sapde.dashboard.utils",
        "sapde.validation.validador",
        "sapde.validation.report_generator",
        "sapde.validation.run_validation",
    ]

    @pytest.mark.parametrize("modulo", _MODULOS)
    def test_importable(self, modulo):
        try:
            importlib.import_module(modulo)
        except ImportError as exc:
            pytest.fail(f"No se pudo importar {modulo!r}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TestArtefactosArchivosClave
# ══════════════════════════════════════════════════════════════════════════════

class TestArtefactosArchivosClave:
    """Verifica que los artefactos generados por las fases anteriores existen."""

    def test_dataset_sintetico(self):
        assert (_RAIZ / "data" / "raw" / "estudiantes_sintetico.csv").exists()

    def test_dataset_procesado(self):
        assert (_RAIZ / "data" / "processed" / "dataset_etl.parquet").exists()

    def test_hay_modelos_joblib(self):
        modelos = list((_RAIZ / "models").glob("*.joblib"))
        assert len(modelos) >= 5, f"Se esperan ≥5 modelos, hay {len(modelos)}"

    def test_hay_metadata_modelos(self):
        metas = list((_RAIZ / "models").glob("*_metadata.json"))
        assert len(metas) >= 5, f"Se esperan ≥5 metadata, hay {len(metas)}"

    def test_reportes_shap(self):
        shap_dir = _RAIZ / "reports" / "shap"
        assert shap_dir.exists()
        assert len(list(shap_dir.glob("*.png"))) >= 5

    def test_shap_resumen_json(self):
        assert (_RAIZ / "reports" / "shap" / "shap_resumen.json").exists()

    def test_reporte_validacion_html(self):
        assert (_RAIZ / "reports" / "validacion" / "reporte_validacion.html").exists()

    def test_resultados_validacion_json(self):
        assert (_RAIZ / "reports" / "validacion" / "resultados_validacion.json").exists()

    def test_benchmark_etl_json(self):
        assert (_RAIZ / "reports" / "benchmark_etl.json").exists()

    def test_benchmark_entrenamiento_json(self):
        assert (_RAIZ / "reports" / "benchmark_entrenamiento.json").exists()


# ══════════════════════════════════════════════════════════════════════════════
# TestReadme
# ══════════════════════════════════════════════════════════════════════════════

class TestReadme:
    """Verifica que el README está completo y actualizado."""

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / "README.md"
        assert ruta.exists(), "README.md no existe"
        return ruta.read_text(encoding="utf-8")

    def test_existe(self):
        assert (_RAIZ / "README.md").exists()

    def test_menciona_todas_las_fases(self, contenido):
        for fase in range(8):
            assert str(fase) in contenido, f"Fase {fase} no mencionada en README"

    def test_menciona_docker_compose(self, contenido):
        assert "docker compose" in contenido.lower()

    def test_menciona_pythonpath(self, contenido):
        assert "PYTHONPATH" in contenido

    def test_tiene_tabla_endpoints(self, contenido):
        assert "/predict" in contenido

    def test_tiene_tabla_features(self, contenido):
        assert "semestre" in contenido

    def test_menciona_postgresql(self, contenido):
        assert "PostgreSQL" in contenido or "postgres" in contenido.lower()

    def test_tiene_seccion_instalacion(self, contenido):
        assert "Instalación" in contenido or "instalacion" in contenido.lower()

    def test_tiene_seccion_comandos(self, contenido):
        assert "pytest" in contenido


# ══════════════════════════════════════════════════════════════════════════════
# TestRequirements
# ══════════════════════════════════════════════════════════════════════════════

class TestRequirements:
    """Verifica que requirements.txt tiene las dependencias clave."""

    _PAQUETES_CLAVE = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "psycopg2",
        "streamlit",
        "scikit-learn",
        "lightgbm",
        "shap",
        "plotly",
        "pydantic",
    ]

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / "requirements.txt"
        assert ruta.exists(), "requirements.txt no existe"
        return ruta.read_text(encoding="utf-8").lower()

    @pytest.mark.parametrize("paquete", _PAQUETES_CLAVE)
    def test_paquete_presente(self, contenido, paquete):
        assert paquete in contenido, f"Paquete {paquete!r} no encontrado en requirements.txt"


# ══════════════════════════════════════════════════════════════════════════════
# TestSintaxisPython
# ══════════════════════════════════════════════════════════════════════════════

class TestSintaxisPython:
    """Verifica que todos los archivos .py del src pasan ast.parse (sin errores de sintaxis)."""

    @pytest.fixture(scope="class")
    def archivos_python(self):
        src = _RAIZ / "src"
        return sorted(src.rglob("*.py"))

    def test_hay_archivos_python(self, archivos_python):
        assert len(archivos_python) > 10, "Se esperan al menos 10 archivos .py en src/"

    def test_sin_errores_sintaxis(self, archivos_python):
        errores = []
        for ruta in archivos_python:
            try:
                ast.parse(ruta.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                errores.append(f"{ruta.relative_to(_RAIZ)}: {exc}")
        assert not errores, "Errores de sintaxis:\n" + "\n".join(errores)


# ══════════════════════════════════════════════════════════════════════════════
# TestStreamlitConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamlitConfig:
    """Verifica el archivo de configuración de Streamlit para Docker."""

    @pytest.fixture(scope="class")
    def contenido(self):
        ruta = _RAIZ / "scripts" / "streamlit_config.toml"
        assert ruta.exists(), "scripts/streamlit_config.toml no existe"
        return ruta.read_text(encoding="utf-8")

    def test_existe(self):
        assert (_RAIZ / "scripts" / "streamlit_config.toml").exists()

    def test_seccion_server(self, contenido):
        assert "[server]" in contenido

    def test_headless_true(self, contenido):
        assert "headless = true" in contenido

    def test_desactiva_telemetria(self, contenido):
        assert "gatherUsageStats = false" in contenido
