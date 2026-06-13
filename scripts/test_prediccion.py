"""
Valores de demostración validados contra el modelo entrenado.
Ejecutar: python scripts/test_prediccion.py
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from sapde.models.artifacts import mejor_modelo
from sapde.features.engineering import agregar_features_avanzadas
from sapde.dashboard.utils import FEATURE_NAMES
from pathlib import Path

pipeline, meta = mejor_modelo(Path("models"))
umbral = meta.get("umbral", 0.5)
print(f"Modelo: {meta['nombre']} v{meta['version']}  umbral: {umbral:.4f}")
print(f"Niveles:  bajo <20%  |  moderado 20-{umbral*100:.1f}%  |  alto {umbral*100:.1f}-70%  |  muy_alto >=70%\n")

def predecir(datos, label):
    df = pd.DataFrame([{**datos, "student_id": "demo", "deserta": 0}])
    df = agregar_features_avanzadas(df)
    X = df[FEATURE_NAMES].values.astype(float)
    proba = pipeline.predict_proba(X)[0, 1]
    if proba >= 0.70:     nivel = "MUY ALTO"
    elif proba >= umbral: nivel = "ALTO"
    elif proba >= 0.20:   nivel = "MODERADO"
    else:                 nivel = "BAJO"
    print(f"{label:22s}  {proba*100:5.1f}%  ->  {nivel}")

# ── Valores extraídos de ejemplos reales del dataset entrenado ──────────────

# MUY ALTO: semestre avanzado, bajo promedio este semestre, avance por debajo de lo esperado
predecir({
    "semestre": 9, "promedio_semestre": 2.18, "promedio_acumulado": 3.02,
    "tendencia_calificaciones": -0.05, "creditos_matriculados": 21, "creditos_aprobados": 10,
    "porcentaje_avance_real": 60.49, "num_asignaturas_reprobadas": 2, "tasa_reprobacion": 0.43,
    "semestres_cursados": 9, "interrupciones_previas": 0,
    "ratio_aprobacion": 10/21, "deficit_creditos": 11.0,
    "score_riesgo_bruto": 0.4*(5-3.02)/4 + 0.3*0.43 + 0.2*(1-60.49/100) + 0.0,
    "anio_cohorte": 2019, "semestre_cohorte": 2,
}, "MUY ALTO (video)")

# ALTO: semestre 6, promedio bajo, tendencia negativa
predecir({
    "semestre": 6, "promedio_semestre": 2.54, "promedio_acumulado": 3.19,
    "tendencia_calificaciones": -0.23, "creditos_matriculados": 21, "creditos_aprobados": 14,
    "porcentaje_avance_real": 41.98, "num_asignaturas_reprobadas": 2, "tasa_reprobacion": 0.43,
    "semestres_cursados": 6, "interrupciones_previas": 0,
    "ratio_aprobacion": 14/21, "deficit_creditos": 7.0,
    "score_riesgo_bruto": 0.4*(5-3.19)/4 + 0.3*0.43 + 0.2*(1-41.98/100) + 0.0,
    "anio_cohorte": 2023, "semestre_cohorte": 2,
}, "ALTO (video)")

# MODERADO: semestre 5, buen promedio pero algo de rezago
predecir({
    "semestre": 5, "promedio_semestre": 3.72, "promedio_acumulado": 3.63,
    "tendencia_calificaciones": 0.08, "creditos_matriculados": 15, "creditos_aprobados": 12,
    "porcentaje_avance_real": 38.27, "num_asignaturas_reprobadas": 1, "tasa_reprobacion": 0.20,
    "semestres_cursados": 5, "interrupciones_previas": 0,
    "ratio_aprobacion": 12/15, "deficit_creditos": 3.0,
    "score_riesgo_bruto": 0.4*(5-3.63)/4 + 0.3*0.20 + 0.2*(1-38.27/100) + 0.0,
    "anio_cohorte": 2019, "semestre_cohorte": 2,
}, "MODERADO (video)")

# BAJO: semestre 2, promedio aceptable, sin reprobadas
predecir({
    "semestre": 2, "promedio_semestre": 2.74, "promedio_acumulado": 3.00,
    "tendencia_calificaciones": -0.52, "creditos_matriculados": 15, "creditos_aprobados": 12,
    "porcentaje_avance_real": 12.96, "num_asignaturas_reprobadas": 0, "tasa_reprobacion": 0.0,
    "semestres_cursados": 2, "interrupciones_previas": 0,
    "ratio_aprobacion": 12/15, "deficit_creditos": 3.0,
    "score_riesgo_bruto": 0.4*(5-3.00)/4 + 0.0 + 0.2*(1-12.96/100) + 0.0,
    "anio_cohorte": 2019, "semestre_cohorte": 1,
}, "BAJO (video)")
