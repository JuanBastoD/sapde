# SAPDE — Sistema de Analítica Predictiva de Deserción Estudiantil

**Programa:** Ingeniería de Sistemas — Universidad de Pamplona (sede Villa del Rosario)  
**Python:** 3.12 | **Plataforma:** Windows 10/11 + PowerShell  
**Tests:** 187 pasando | **Estado:** Completo (7 fases)

---

## Descripción

SAPDE predice la probabilidad de deserción estudiantil por semestre usando cinco técnicas de aprendizaje automático (SVM, Random Forest, LightGBM, MLP, ANFIS), con interpretabilidad SHAP, API REST y dashboard interactivo. Desarrollado como proyecto estudiantil para la Facultad de ingenierías y arquitecturas.

### Objetivos específicos

| OE | Descripción | Módulo |
|----|-------------|--------|
| OE1 | ETL paralelo con OpenMP + análisis de patrones (EDA) | `etl_cpp/`, `src/sapde/etl/`, `notebooks/` |
| OE2 | Selección y evaluación de 5 técnicas ML | `src/sapde/models/` |
| OE3 | Paralelización en entrenamiento e inferencia | `src/sapde/models/train.py` |
| OE4 | Prototipo de visualización interactiva | `src/sapde/dashboard/`, `src/sapde/api/` |
| OE5 | Validación estadística + benchmark paralelo | `src/sapde/validation/`, `reports/` |

---

## Resultados principales

| Modelo | AUC-ROC | IC 95% (bootstrap) | Recall | F1 |
|--------|---------|-------------------|--------|-----|
| SVM | 0.836 | [0.795, 0.872] | — | — |
| RandomForest | 0.835 | [0.801, 0.869] | — | — |
| LightGBM | 0.833 | [0.799, 0.865] | — | — |
| ANFIS | 0.828 | [0.791, 0.864] | — | — |
| MLP | 0.813 | [0.777, 0.853] | — | — |

Top features (SHAP): `semestre`, `repro_x_deficit`, `num_asignaturas_reprobadas`, `promedio_acumulado`

---

## Instalación en Windows (PowerShell)

### Requisitos previos

- Python 3.12: <https://www.python.org/downloads/release/python-3120/>
- Git: <https://git-scm.com/>
- PostgreSQL 16+ (opcional, usa SQLite como fallback automático)
- Docker Desktop (opcional, para contenedores): <https://www.docker.com/products/docker-desktop/>

### Pasos

```powershell
# 1. Navegar al repositorio
cd C:\ruta\al\proyecto\sapde

# 2. Crear entorno virtual e instalar dependencias
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Configurar variables de entorno
Copy-Item .env.example .env
# Editar .env con tu configuración de PostgreSQL

# 4. Generar dataset sintético
$env:PYTHONPATH = "src"
py -3.12 -m sapde.data.synthetic

# 5. Ejecutar ETL y entrenar modelos
py -3.12 -m sapde.etl.pipeline
py -3.12 -m sapde.models.train

# 6. Verificar con todos los tests
py -3.12 -m pytest tests\ -v
```

### Instalación con Docker

```powershell
# Copiar y editar variables de entorno
Copy-Item .env.example .env
# Editar .env con contraseñas seguras

# Levantar todos los servicios
docker compose up --build

# Solo base de datos (para desarrollo local)
docker compose up db

# API + DB (sin dashboard)
docker compose up db backend

# Detener y limpiar
docker compose down
docker compose down -v    # incluye borrar volumen de datos PostgreSQL
```

Servicios disponibles tras `docker compose up --build`:

| Servicio | URL | Descripción |
|----------|-----|-------------|
| API REST | <http://localhost:8000> | FastAPI |
| Docs API | <http://localhost:8000/docs> | Swagger UI |
| Dashboard | <http://localhost:8501> | Streamlit |
| PostgreSQL | `localhost:5432` | Base de datos |

---

## Estructura del proyecto

```
sapde/
├── data/
│   ├── raw/                  → CSV sintético (o institucional)
│   ├── interim/              → Benchmarks ETL intermedios
│   └── processed/            → Parquet procesado por ETL
├── etl_cpp/                  → ETL paralelo en C++/OpenMP (OE1)
│   ├── src/etl_main.cpp
│   └── CMakeLists.txt
├── models/                   → Artefactos entrenados (.joblib + metadata JSON)
├── reports/
│   ├── shap/                 → Gráficas SHAP por modelo
│   ├── validacion/           → Reporte HTML + JSON de validación
│   └── *.png, *.json         → Benchmarks, curvas ROC/PR, EDA
├── scripts/
│   ├── setup.ps1             → Setup automático (Windows)
│   ├── run.ps1               → Comandos de ejecución
│   ├── init_db.sql           → Inicialización PostgreSQL
│   └── streamlit_config.toml → Configuración Streamlit (Docker)
├── src/sapde/
│   ├── config.py             → Configuración centralizada (Pydantic-settings)
│   ├── data/synthetic.py     → Generador de datos sintéticos
│   ├── etl/                  → ETL Python (pipeline + fallback joblib)
│   ├── features/             → Ingeniería de variables (16 base + 9 avanzadas)
│   ├── models/               → Entrenamiento, artefactos, comparación
│   ├── explain/              → SHAP (TreeExplainer + KernelExplainer)
│   ├── eval/                 → Métricas y benchmark
│   ├── db/                   → SQLAlchemy ORM + CRUD + sesión
│   ├── api/                  → FastAPI (rutas, schemas, dependencias)
│   ├── dashboard/            → Streamlit multi-página (4 páginas)
│   └── validation/           → Validación estadística + reporte HTML
├── tests/
│   ├── test_fase0.py         → Config, dataset sintético
│   ├── test_fase1.py         → ETL, EDA, benchmark
│   ├── test_fase2.py         → Modelos ML, artefactos
│   ├── test_fase3.py         → SHAP, interpretabilidad
│   ├── test_fase4.py         → API FastAPI, BD
│   ├── test_fase5.py         → Dashboard Streamlit
│   ├── test_fase6.py         → Validación estadística
│   └── test_fase7.py         → Integración Docker, README, cobertura
├── Dockerfile.backend        → Imagen Docker para API
├── Dockerfile.dashboard      → Imagen Docker para Dashboard
├── docker-compose.yml        → Orquestación de servicios
├── .dockerignore
├── .env.example              → Plantilla de variables de entorno
├── pyproject.toml            → Configuración del proyecto
├── requirements.txt          → Dependencias de producción
└── requirements-dev.txt      → Dependencias de desarrollo
```

---

## Comandos frecuentes

```powershell
# Siempre activar el entorno y establecer PYTHONPATH primero
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

# Regenerar dataset sintético
py -3.12 -m sapde.data.synthetic

# Ejecutar ETL (fallback Python)
py -3.12 -m sapde.etl.pipeline

# Entrenar todos los modelos
py -3.12 -m sapde.models.train

# Calcular SHAP para todos los modelos
py -3.12 -m sapde.explain.shap_explainer

# Ejecutar validación estadística completa (genera reporte HTML)
py -3.12 -m sapde.validation.run_validation

# Lanzar API (desarrollo)
uvicorn sapde.api.main:app --reload --host 0.0.0.0 --port 8000

# Lanzar dashboard
streamlit run src/sapde/dashboard/app.py

# Tests completos
py -3.12 -m pytest tests\ -v

# Tests con cobertura
py -3.12 -m pytest tests\ --cov=sapde --cov-report=term-missing

# Linting
ruff check src\
```

---

## API REST — Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio y BD |
| `GET` | `/health/model` | Información del modelo activo |
| `POST` | `/predict` | Predicción de riesgo para un estudiante |
| `POST` | `/predict?shap=true` | Predicción + explicación SHAP |
| `GET` | `/students/at-risk` | Lista de estudiantes en riesgo alto/muy alto |
| `GET` | `/students/{id}/explanation` | Última predicción SHAP de un estudiante |

### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "EST_001",
    "semestre": 3,
    "promedio_semestre": 3.1,
    "promedio_acumulado": 3.3,
    "tendencia_calificaciones": -0.2,
    "creditos_matriculados": 18,
    "creditos_aprobados": 12,
    "porcentaje_avance_real": 35.0,
    "num_asignaturas_reprobadas": 2,
    "tasa_reprobacion": 0.33,
    "semestres_cursados": 3,
    "interrupciones_previas": 0,
    "semestre_cohorte": 1
  }'
```

---

## Features del modelo

El modelo usa **25 variables**:

**Base (ETL — 16):** `semestre`, `promedio_semestre`, `promedio_acumulado`, `tendencia_calificaciones`, `creditos_matriculados`, `creditos_aprobados`, `porcentaje_avance_real`, `num_asignaturas_reprobadas`, `tasa_reprobacion`, `semestres_cursados`, `interrupciones_previas`, `semestre_cohorte`

**Avanzadas (ingeniería — 9):** `deficit_creditos`, `ratio_aprobacion`, `eficiencia_acumulada`, `carga_academica_norm`, `riesgo_academico_score`, `repro_x_deficit`, `tendencia_negativa`, `semestre_critico`, `sobrecarga`

---

## Punto de integración de datos reales

Para usar datos institucionales reales, reemplazar `data/raw/estudiantes_sintetico.csv` con un CSV que tenga las siguientes columnas mínimas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `student_id` | str | Identificador único del estudiante |
| `cohorte` | str | Año y semestre de ingreso (ej. `2021-I`) |
| `semestre` | int | Número de semestre cursado |
| `promedio_semestre` | float | Nota promedio del semestre (0–5) |
| `promedio_acumulado` | float | Promedio histórico acumulado (0–5) |
| `tendencia_calificaciones` | float | Pendiente de notas en el tiempo |
| `creditos_matriculados` | int | Créditos inscritos el semestre |
| `creditos_aprobados` | int | Créditos aprobados el semestre |
| `porcentaje_avance_real` | float | % avance sobre créditos del programa |
| `num_asignaturas_reprobadas` | int | Asignaturas reprobadas el semestre |
| `tasa_reprobacion` | float | Fracción de asignaturas reprobadas (0–1) |
| `semestres_cursados` | int | Semestres cursados hasta la fecha |
| `interrupciones_previas` | int | Veces que dejó de matricularse |
| `deserta` | int | Etiqueta: 0 = permanece, 1 = deserta |

La función `cargar_datos_reales(ruta)` en `src/sapde/data/synthetic.py` ya está preparada para recibir esta ruta.

---

## Fases de desarrollo

| Fase | Tests | Descripción |
|------|-------|-------------|
| 0 | ✅ | Andamiaje, config (Pydantic-settings), dataset sintético (10 618 filas, 1 500 estudiantes, 28.4% desertores) |
| 1 | ✅ | ETL paralelo C++/OpenMP (fallback joblib), EDA con 5 gráficas, benchmark speedup |
| 2 | ✅ | 5 modelos ML + SMOTE + búsqueda de hiperparámetros (Optuna). Artefactos `.joblib` + metadata JSON |
| 3 | ✅ | Interpretabilidad SHAP: TreeExplainer (RF, LGB), KernelExplainer (SVM, MLP, ANFIS). 32 archivos en `reports/shap/` |
| 4 | ✅ | API FastAPI + SQLAlchemy 2 + PostgreSQL (fallback SQLite automático). 5 endpoints REST |
| 5 | ✅ | Dashboard Streamlit 4 páginas: inicio, predicción, estudiantes en riesgo, métricas de modelos |
| 6 | ✅ | Validación estadística: Bootstrap IC (500 iter), 5-fold CV estratificada, curvas calibración. Reporte HTML autocontenido |
| 7 | ✅ | Docker Compose (db + backend + dashboard), `.dockerignore`, tests integración, README completo |

**Total: 200+ tests pasando**

---

## Configuración PostgreSQL (local)

Si PostgreSQL 18 está instalado en Windows, crear usuario y base de datos:

```python
# Ejecutar desde Python (psql.exe puede estar bloqueado por AppControl)
import psycopg2
conn = psycopg2.connect(host="127.0.0.1", port=5432,
                        dbname="postgres", user="postgres", password="postgres")
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE USER sapde_user WITH PASSWORD 'sapde_password';")
cur.execute("CREATE DATABASE sapde_db OWNER sapde_user;")
conn.close()
```

**Nota importante:** En Windows, PostgreSQL resuelve `localhost` como `::1` (IPv6) lo que puede fallar. Usar `POSTGRES_HOST=127.0.0.1` en `.env`.

---

## Licencia

MIT — Proyecto académico, Universidad de Pamplona.
