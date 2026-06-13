# Guion de demostración — SAPDE
**Duración estimada:** 12–15 minutos  
**Herramientas abiertas antes de grabar:** VS Code, PowerShell, navegador en `localhost:8000/docs` y `localhost:8501`

---

## Intro (0:00 – 0:45)

**Pantalla:** README.md en GitHub → `https://github.com/JuanBastoD/sapde`

> "Buenos días / tardes. Voy a presentar SAPDE: Sistema de Analítica Predictiva de Deserción Estudiantil, desarrollado para el programa de Ingeniería de Sistemas de la Universidad de Pamplona, sede Villa del Rosario.
>
> El sistema predice la probabilidad de que un estudiante abandone sus estudios en el semestre siguiente, usando cinco técnicas de aprendizaje automático. Voy a demostrar el cumplimiento de cada uno de los cinco objetivos específicos del proyecto."

**Pantalla:** Tabla de OE del README.

---

## OE1 — ETL paralelo con OpenMP + análisis de patrones (1:00 – 3:30)

### 1.1 Mostrar el código C++

**Pantalla:** Abrir `etl_cpp/src/etl_main.cpp` en VS Code

> "El primer objetivo fue implementar un módulo ETL de alto rendimiento usando C++ con OpenMP para procesamiento paralelo. Aquí vemos el pipeline que lee el CSV institucional, limpia valores atípicos y calcula variables derivadas como la tasa de reprobación y la tendencia de calificaciones, usando directivas `#pragma omp parallel for` para distribuir el trabajo entre hilos."

**Pantalla:** Scroll al bloque `#pragma omp` en el archivo.

### 1.2 Ejecutar el ETL

**Pantalla:** PowerShell

```powershell
cd C:\Users\Juanes\Documents\Universidad\9no\fcpd\Proyecto\sapde
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m sapde.etl.pipeline
```

> "El módulo Python detecta si la librería C++ compilada está disponible y hace fallback automático a la implementación en Python con joblib si no lo está. En Windows, usamos el fallback que replica el mismo algoritmo."

### 1.3 Benchmark de speedup ETL

**Pantalla:** PowerShell — mostrar el JSON

```powershell
Get-Content reports\benchmark_etl.json | .\.venv\Scripts\python.exe -c "import sys,json; [print(r) for r in json.load(sys.stdin)]"
```

> "El benchmark muestra el tiempo de procesamiento de los 10.618 registros con 1, 2, 4 y 8 hilos. Con 1 hilo el ETL tarda 0.039 segundos. La comparación de rendimiento entre la versión secuencial y paralela está documentada en el reporte de validación."

### 1.4 EDA — Análisis de patrones

**Pantalla:** Abrir `reports/eda_resumen.txt` en VS Code, luego las imágenes de EDA

> "El análisis exploratorio identificó los patrones clave: los estudiantes en riesgo tienden a reprobar más asignaturas, tienen promedio acumulado menor a 3.2 y presentan interrupciones previas. Estas son las gráficas generadas: distribución de variables, correlaciones y tasa de deserción por cohorte."

**Pantalla:** Abrir `reports/eda_tasa_desercion_cohorte.png` y `reports/eda_correlacion.png`

---

## OE2 — Selección y evaluación de técnicas ML (3:30 – 6:00)

### 2.1 Código de los 5 modelos

**Pantalla:** Abrir `src/sapde/models/pipeline_ml.py` en VS Code

> "El segundo objetivo fue seleccionar y evaluar cinco técnicas de machine learning: SVM, Random Forest, LightGBM, MLP y ANFIS. Cada modelo se envuelve en un pipeline scikit-learn que incluye escalado con StandardScaler y balanceo de clases con SMOTE, dado que la tasa de deserción a nivel de registro es del 4%."

### 2.2 Resultados comparativos

**Pantalla:** PowerShell

```powershell
Get-Content reports\comparacion_modelos.csv
```

> "Después del entrenamiento con búsqueda de hiperparámetros usando Optuna, los resultados son:"

| Modelo | AUC-ROC | Recall | F1 |
|--------|---------|--------|----|
| **SVM** | **0.8382** | — | — |
| RandomForest | 0.8352 | — | — |
| LightGBM | 0.8338 | — | — |
| ANFIS | 0.8280 | — | — |
| MLP | 0.8188 | — | — |

> "El SVM obtiene el mayor AUC-ROC con 0.838. Los cinco modelos superan 0.80, lo que indica discriminación sólida."

### 2.3 Curvas ROC y PR

**Pantalla:** Abrir `reports/curvas_roc.png` y `reports/curvas_pr.png`

> "Las curvas ROC y Precision-Recall confirman el desempeño. Se eligió el umbral de clasificación mediante el criterio de Youden's J para maximizar la detección de desertores sin generar demasiadas falsas alarmas."

### 2.4 Matrices de confusión

**Pantalla:** Abrir `reports/matrices_confusion.png`

> "Las matrices de confusión muestran el balance entre sensibilidad y especificidad para cada modelo con su umbral óptimo."

### 2.5 Interpretabilidad SHAP

**Pantalla:** Abrir `reports/shap/RandomForest_summary.png`

> "Para cumplir con la interpretabilidad, se calcularon los valores SHAP para cada modelo. Las variables más importantes son: el semestre cursado, la interacción entre reprobaciones y déficit de créditos, el número de asignaturas reprobadas y el promedio acumulado. Esto permite al orientador universitario entender por qué el sistema marca a un estudiante como en riesgo."

**Pantalla:** Abrir `reports/shap/comparativa_modelos_shap.png`

> "Esta gráfica comparativa muestra que los cinco modelos coinciden en las variables más relevantes, lo que valida la robustez de las features."

---

## OE3 — Paralelización en entrenamiento (6:00 – 7:30)

### 3.1 Benchmark de entrenamiento

**Pantalla:** PowerShell

```powershell
Get-Content reports\benchmark_entrenamiento.json | .\.venv\Scripts\python.exe -c "import sys,json; [print(r) for r in json.load(sys.stdin)]"
```

> "El tercer objetivo fue medir el impacto de la paralelización en el entrenamiento. Los modelos que soportan múltiples hilos son Random Forest y LightGBM."
>
> - **LightGBM** pasa de 1.30 s (1 hilo) a 0.73 s (2 hilos) → **speedup 1.77×**
> - **Random Forest** pasa de 4.38 s (1 hilo) a 3.28 s (2 hilos) → **speedup 1.33×**
>
> "SVM y MLP son inherentemente secuenciales, mientras que ANFIS usa la implementación scikit-fuzzy sin soporte de hilos."

### 3.2 Código del entrenamiento paralelo

**Pantalla:** Abrir `src/sapde/models/train.py` en VS Code, buscar `n_jobs`

> "El parámetro `n_jobs` se controla desde la configuración centralizada. En producción se usa `n_jobs=-1` para emplear todos los núcleos disponibles."

---

## OE4 — Prototipo de visualización interactiva (7:30 – 11:00)

### 4.1 API REST — Swagger UI

**Pantalla:** Navegador en `http://localhost:8000/docs`

> "El cuarto objetivo fue un prototipo de visualización interactiva. La capa de acceso a datos es una API REST construida con FastAPI, documentada automáticamente con Swagger UI."

#### Demo: GET /health

**Pantalla:** Clic en `GET /health` → Try it out → Execute

> "El endpoint de salud confirma que el servicio está activo, el modelo cargado es SVM_v5 y la base de datos PostgreSQL está conectada."

#### Demo: POST /predict — caso ALTO (69.7%)

**Pantalla:** Clic en `POST /predict` → Try it out → pegar este JSON en el body:

```json
{
  "student_id": "EST_DEMO_ALTO",
  "semestre": 6,
  "promedio_semestre": 2.54,
  "promedio_acumulado": 3.19,
  "tendencia_calificaciones": -0.23,
  "creditos_matriculados": 21,
  "creditos_aprobados": 14,
  "porcentaje_avance_real": 41.98,
  "num_asignaturas_reprobadas": 2,
  "tasa_reprobacion": 0.43,
  "semestres_cursados": 6,
  "interrupciones_previas": 0,
  "semestre_cohorte": 2
}
```

> "Para un estudiante de sexto semestre con promedio 2.54 este semestre, avance del 42% del programa y tendencia negativa, el sistema predice riesgo ALTO con el 69.7% de probabilidad."

**Pantalla:** Mostrar la respuesta — probabilidad, nivel de riesgo, umbral.

#### Demo: POST /predict?shap=true

> "Con el parámetro `shap=true`, la respuesta incluye además los valores SHAP: cuáles variables están empujando la predicción hacia riesgo alto y cuáles la mitigan."

#### Demo: GET /students/at-risk

> "Este endpoint lista todos los estudiantes clasificados como riesgo alto o muy alto, con su probabilidad estimada y las recomendaciones de intervención generadas automáticamente."

---

### 4.2 Dashboard Streamlit

**Pantalla:** Navegador en `http://localhost:8501`

> "El dashboard interactivo está diseñado para el orientador universitario, sin necesidad de conocimientos técnicos."

#### Página 1 — Inicio

> "La página de inicio muestra métricas globales: total de predicciones, distribución por nivel de riesgo y un histograma de probabilidades."

#### Página 2 — Predicción individual

**Pantalla:** Clic en "Prediccion Individual" en el menú lateral

> "El orientador ingresa los datos académicos del semestre del estudiante usando controles visuales. El sistema calcula la predicción en tiempo real."

**Pantalla:** Llenar el formulario con estos valores (dan MUY ALTO — 96.6%):

| Campo | Valor |
|-------|-------|
| Semestre | **9** |
| Promedio semestre | **2.18** |
| Promedio acumulado | **3.02** |
| Tendencia notas | **-0.05** |
| Créditos matriculados | **21** |
| Créditos aprobados | **10** |
| % Avance real | **60.49** |
| Asignaturas reprobadas | **2** |
| Tasa reprobación | **0.43** |
| Semestres cursados | **9** |
| Interrupciones previas | **0** |
| Año de ingreso | **2019** |
| Semestre de ingreso | **2** |

> "El gauge muestra 96.6% — riesgo MUY ALTO. Un estudiante en semestre 9 con solo el 60% de avance del programa y promedio bajo este semestre es el perfil típico que el modelo identifica como en peligro inminente de desertar. El SHAP explica cuáles variables lo llevan a este nivel."

**Para mostrar ALTO (69.7%):** cambiar a semestre=6, promedio_semestre=2.54, promedio_acumulado=3.19, creditos_aprobados=14, porcentaje_avance_real=41.98, tasa_reprobacion=0.43, semestres_cursados=6, semestre_cohorte=2

**Para mostrar MODERADO (22.5%):** semestre=5, promedio_semestre=3.72, promedio_acumulado=3.63, creditos_matriculados=15, creditos_aprobados=12, porcentaje_avance_real=38.27, num_asignaturas_reprobadas=1, tasa_reprobacion=0.20, semestres_cursados=5, anio_cohorte=2019, semestre_cohorte=2

**Para mostrar BAJO (8.9%):** semestre=2, promedio_semestre=2.74, promedio_acumulado=3.02, creditos_matriculados=15, creditos_aprobados=12, porcentaje_avance_real=12.96, num_asignaturas_reprobadas=1, tasa_reprobacion=0.20, semestres_cursados=2, anio_cohorte=2023, semestre_cohorte=2

#### Página 3 — Estudiantes en riesgo

**Pantalla:** Clic en "Estudiantes en Riesgo"

> "Esta página permite al equipo de bienestar priorizar intervenciones. Se puede filtrar por nivel de riesgo, buscar estudiantes específicos y descargar la lista en CSV para gestionar el seguimiento."

**Pantalla:** Mostrar el scatter plot y hacer clic en "Descargar CSV"

#### Página 4 — Metricas de modelos

**Pantalla:** Clic en "Metricas de Modelos"

> "La última página compara los cinco modelos: tabla de métricas, gráfico radar que resume AUC, Recall, F1 y Precisión simultáneamente, y el mapa de calor SHAP que confirma qué variables importan en cada modelo."

---

## OE5 — Validación integral + benchmark (11:00 – 13:30)

### 5.1 Abrir el reporte HTML

**Pantalla:** Abrir `reports/validacion/reporte_validacion.html` en el navegador

> "El quinto objetivo fue la validación estadística rigurosa. El reporte autocontenido documenta todos los resultados."

### 5.2 Intervalos de confianza Bootstrap

**Pantalla:** Scroll a la sección Bootstrap del reporte

> "Para cada modelo se calcularon intervalos de confianza al 95% con 500 iteraciones de bootstrap sobre el conjunto de test, sin reentrenamiento. Esto da una medida honesta de la variabilidad de las métricas:"
>
> - SVM: AUC = 0.836 **[0.795 – 0.872]**
> - RandomForest: AUC = 0.835 **[0.801 – 0.869]**
> - LightGBM: AUC = 0.833 **[0.799 – 0.865]**
> - ANFIS: AUC = 0.828 **[0.791 – 0.864]**
> - MLP: AUC = 0.813 **[0.777 – 0.853]**
>
> "Los intervalos no se solapan con AUC < 0.75, lo que confirma que todos los modelos tienen poder discriminativo estadísticamente significativo."

### 5.3 Validación cruzada 5-fold

**Pantalla:** Scroll a sección CV del reporte

> "La validación cruzada estratificada en 5 folds, aplicada sobre el pipeline completo incluyendo SMOTE, confirma que los modelos no están sobreajustados. El AUC en CV es consistente con el del test set, con desviación estándar menor a 0.02 en todos los casos."

### 5.4 Curvas de calibración

**Pantalla:** Scroll a curvas de calibración

> "Las curvas de calibración muestran qué tan bien calibradas están las probabilidades. Random Forest y MLP tienen el mejor ECE (Expected Calibration Error), lo que significa que cuando el modelo dice '70% de riesgo', efectivamente cerca del 70% de esos estudiantes desertan."

### 5.5 Gráficas de speedup

**Pantalla:** Scroll a sección de benchmark del reporte

> "El reporte también incluye las gráficas de speedup del ETL y del entrenamiento paralelo, cumpliendo el objetivo de comparar el rendimiento secuencial versus paralelo documentado en OE5."

---

## Cierre (13:30 – 14:30)

**Pantalla:** Volver al README en GitHub

> "En resumen, SAPDE cumple los cinco objetivos específicos:
>
> - **OE1:** ETL implementado en C++ con OpenMP y fallback Python/joblib, con benchmark de speedup documentado y EDA con cinco gráficas de análisis de patrones.
>
> - **OE2:** Cinco técnicas evaluadas — SVM, Random Forest, LightGBM, MLP y ANFIS — con AUC entre 0.81 y 0.84, e interpretabilidad SHAP para cada una.
>
> - **OE3:** Paralelización medida en entrenamiento: LightGBM logra speedup de 1.77× con 2 hilos, Random Forest 1.33×.
>
> - **OE4:** Prototipo interactivo completo: API REST con FastAPI y base de datos PostgreSQL, más dashboard Streamlit de cuatro páginas accesible para usuarios no técnicos.
>
> - **OE5:** Validación estadística integral: Bootstrap IC al 95%, validación cruzada 5-fold estratificada, curvas de calibración y reporte HTML autocontenido.
>
> El sistema tiene 294 tests automatizados y está disponible en GitHub: github.com/JuanBastoD/sapde"

**Pantalla:** `https://github.com/JuanBastoD/sapde`

---

## Comandos de referencia rápida

```powershell
# Levantar el sistema completo (si no está corriendo)
cd C:\Users\Juanes\Documents\Universidad\9no\fcpd\Proyecto\sapde
$env:PYTHONPATH = "src"

# API (en ventana minimizada)
Start-Process .\.venv\Scripts\uvicorn.exe -ArgumentList "sapde.api.main:app","--host","0.0.0.0","--port","8000","--reload" -WindowStyle Minimized

# Dashboard (en ventana minimizada)
Start-Process .\.venv\Scripts\streamlit.exe -ArgumentList "run","src\sapde\dashboard\app.py","--server.port=8501" -WindowStyle Minimized

# Verificar que ambos están vivos
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest http://localhost:8501 -UseBasicParsing | Select-Object StatusCode

# Detener todo
Get-Process uvicorn, streamlit | Stop-Process -Force
```

---

## Checklist antes de grabar

- [ ] API corriendo y respondiendo en `localhost:8000/health`
- [ ] Dashboard corriendo en `localhost:8501`
- [ ] Navegador con dos pestañas: Swagger UI y Dashboard
- [ ] VS Code con `etl_cpp/src/etl_main.cpp` y `src/sapde/models/pipeline_ml.py` abiertos
- [ ] PowerShell en la carpeta del proyecto con `$env:PYTHONPATH = "src"`
- [ ] Carpeta `reports/` abierta en el explorador de Windows para acceder rápido a las imágenes
- [ ] Reporte HTML abierto: `reports/validacion/reporte_validacion.html`
- [ ] Micrófono testeado, notificaciones silenciadas, segunda pantalla si tienes
