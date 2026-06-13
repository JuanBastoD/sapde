-- Script de inicializacion de la base de datos SAPDE
-- Se ejecuta automaticamente al crear el contenedor Docker por primera vez

-- Extensiones utiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- busqueda de texto rapida

-- El esquema completo se crea en Fase 4 via SQLAlchemy/Alembic
-- Este script solo garantiza que las extensiones esten disponibles
