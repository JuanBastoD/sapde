# ============================================================
# SAPDE — Makefile  (alternativa a los scripts PowerShell)
# En Windows usar: nmake o instalar GNU Make via choco/winget
# ============================================================

.PHONY: help setup install datos etl entrenar api dashboard test lint docker-up docker-down

PYTHON := py -3.12

help:
	@echo "Comandos disponibles:"
	@echo "  setup      - Crea entorno virtual e instala dependencias"
	@echo "  install    - Solo instala dependencias en el venv activo"
	@echo "  datos      - Genera dataset sintetico en data/raw/"
	@echo "  etl        - Ejecuta pipeline ETL (C++ o fallback Python)"
	@echo "  entrenar   - Entrena todos los modelos (Fase 2)"
	@echo "  api        - Levanta API FastAPI en http://localhost:8000"
	@echo "  dashboard  - Levanta dashboard Streamlit en http://localhost:8501"
	@echo "  test       - Ejecuta suite de pruebas con pytest"
	@echo "  lint       - Verifica estilo con ruff y black"
	@echo "  docker-up  - Levanta todos los servicios con Docker Compose"
	@echo "  docker-down- Detiene todos los servicios Docker"

setup:
	$(PYTHON) -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\pip install -r requirements-dev.txt
	Copy-Item .env.example .env

install:
	pip install -r requirements-dev.txt

datos:
	$(PYTHON) -m sapde.data.synthetic

etl:
	$(PYTHON) -m sapde.etl.pipeline

entrenar:
	$(PYTHON) -m sapde.models.train

api:
	uvicorn sapde.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --cov=src/sapde --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
