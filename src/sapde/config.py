"""
Configuración centralizada de SAPDE.
Lee variables desde .env (o el entorno del sistema).
"""

import io
import logging
import sys
from pathlib import Path
from functools import lru_cache

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directorio raíz del proyecto (dos niveles arriba de src/sapde/)
_RAIZ = Path(__file__).resolve().parents[2]


class Configuracion(BaseSettings):
    """Variables de configuración del sistema SAPDE."""

    model_config = SettingsConfigDict(
        env_file=_RAIZ / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Base de datos ---
    postgres_user: str = "sapde_user"
    postgres_password: str = "sapde_password"
    postgres_db: str = "sapde_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False

    # --- Dashboard ---
    dashboard_port: int = 8501
    dashboard_host: str = "0.0.0.0"

    # --- Rutas (relativas a la raíz del proyecto) ---
    data_raw_path: Path = Path("data/raw")
    data_interim_path: Path = Path("data/interim")
    data_processed_path: Path = Path("data/processed")
    models_path: Path = Path("models")
    reports_path: Path = Path("reports")

    # --- Machine Learning ---
    random_seed: int = 42
    n_jobs: int = -1
    test_size: float = 0.2
    smote_sampling: float = 0.5

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = ""

    # --- Dataset sintético ---
    synthetic_n_estudiantes: int = 1500
    synthetic_semilla: int = 42

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @field_validator("log_level")
    @classmethod
    def validar_log_level(cls, v: str) -> str:
        niveles_validos = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in niveles_validos:
            raise ValueError(f"log_level debe ser uno de: {niveles_validos}")
        return v_upper

    def rutas_absolutas(self) -> dict[str, Path]:
        """Devuelve las rutas de datos como absolutas respecto a la raíz."""
        return {
            "raw": _RAIZ / self.data_raw_path,
            "interim": _RAIZ / self.data_interim_path,
            "processed": _RAIZ / self.data_processed_path,
            "models": _RAIZ / self.models_path,
            "reports": _RAIZ / self.reports_path,
        }


@lru_cache(maxsize=1)
def obtener_config() -> Configuracion:
    """Devuelve la instancia única de configuración (singleton)."""
    return Configuracion()


def configurar_logging(config: Configuracion | None = None) -> logging.Logger:
    """Configura el sistema de logging para SAPDE."""
    cfg = config or obtener_config()
    nivel = getattr(logging, cfg.log_level, logging.INFO)

    formato = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    # Forzar UTF-8 en la consola de Windows (evita UnicodeEncodeError con cp1252)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, io.UnsupportedOperation):
        pass  # reconfigure no disponible en todos los entornos
    manejadores: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if cfg.log_file:
        log_path = _RAIZ / cfg.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        manejadores.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=nivel,
        format=formato,
        handlers=manejadores,
        force=True,
    )

    # Silenciar loggers ruidosos de librerías externas
    for lib in ["httpx", "urllib3", "asyncio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logging.getLogger("sapde")


# Logger raíz del proyecto
logger = configurar_logging()
