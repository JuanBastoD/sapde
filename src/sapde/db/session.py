"""
Sesión de base de datos para SAPDE.

Intenta conectar a PostgreSQL (config.py). Si falla, cae a SQLite.
"""

import logging
import os

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from sapde.db.models import Base

logger = logging.getLogger("sapde.db.session")

_SQLITE_FALLBACK = "sqlite:///./sapde_dev.db"


def _make_engine(database_url: str | None = None):
    if database_url is None:
        try:
            from sapde.config import obtener_config
            database_url = obtener_config().database_url
        except Exception:
            database_url = _SQLITE_FALLBACK

    # Intenta PostgreSQL; si la conexión falla, usa SQLite
    if not database_url.startswith("sqlite"):
        try:
            engine = create_engine(
                database_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 3},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Conectado a PostgreSQL: %s", database_url.split("@")[-1])
            return engine
        except Exception as exc:
            logger.warning(
                "PostgreSQL no disponible (%s). Usando SQLite: %s",
                exc,
                _SQLITE_FALLBACK,
            )
            database_url = _SQLITE_FALLBACK

    # SQLite: habilita foreign keys
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_fk(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    logger.info("Base de datos SQLite: %s", database_url)
    return engine


# Engine y sesión compartidos (se inicializan al importar el módulo)
engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Crea todas las tablas si no existen."""
    Base.metadata.create_all(bind=engine)
    logger.info("Tablas creadas/verificadas.")


def get_db():
    """Generador FastAPI para inyección de sesión DB."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def override_engine(url: str):
    """
    Reemplaza el engine globalmente.  Útil para tests (SQLite en memoria).
    """
    global engine, SessionLocal
    engine = create_engine(url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
