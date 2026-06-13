"""
Inserta predicciones de prueba en la base de datos usando el modelo real.
Genera 20 estudiantes con perfiles variados (muy_alto, alto, moderado, bajo).

Uso:
    cd <raiz_proyecto>
    .\.venv\Scripts\python.exe scripts/seed_predicciones.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import numpy as np
from sapde.db.session import engine, init_db
from sapde.db.crud import crear_prediccion
from sapde.models.artifacts import mejor_modelo
from sapde.features.engineering import agregar_features_avanzadas
from sqlalchemy.orm import sessionmaker
import pandas as pd

_RAIZ   = Path(__file__).resolve().parents[1]
_MODELS = _RAIZ / "models"

# Perfiles de estudiantes — (descripcion, campos base)
# deficit y score se calculan igual que el ETL / dashboard
PERFILES = [
    # ── MUY ALTO ─────────────────────────────────────────────────────────────
    dict(student_id="EST_2019A_001", semestre=9,  promedio_semestre=2.18, promedio_acumulado=3.02,
         tendencia_calificaciones=-0.05, creditos_matriculados=21, creditos_aprobados=10,
         porcentaje_avance_real=60.49, num_asignaturas_reprobadas=2, tasa_reprobacion=0.43,
         semestres_cursados=9,  interrupciones_previas=0, anio_cohorte=2019, semestre_cohorte=2),

    dict(student_id="EST_2018B_007", semestre=10, promedio_semestre=1.95, promedio_acumulado=2.87,
         tendencia_calificaciones=-0.12, creditos_matriculados=18, creditos_aprobados=7,
         porcentaje_avance_real=65.43, num_asignaturas_reprobadas=3, tasa_reprobacion=0.60,
         semestres_cursados=11, interrupciones_previas=1, anio_cohorte=2018, semestre_cohorte=2),

    dict(student_id="EST_2020A_014", semestre=8,  promedio_semestre=2.30, promedio_acumulado=2.95,
         tendencia_calificaciones=-0.20, creditos_matriculados=21, creditos_aprobados=9,
         porcentaje_avance_real=55.56, num_asignaturas_reprobadas=3, tasa_reprobacion=0.57,
         semestres_cursados=9,  interrupciones_previas=1, anio_cohorte=2020, semestre_cohorte=1),

    dict(student_id="EST_2019B_022", semestre=9,  promedio_semestre=2.45, promedio_acumulado=3.10,
         tendencia_calificaciones=-0.10, creditos_matriculados=18, creditos_aprobados=8,
         porcentaje_avance_real=58.02, num_asignaturas_reprobadas=2, tasa_reprobacion=0.50,
         semestres_cursados=10, interrupciones_previas=0, anio_cohorte=2019, semestre_cohorte=2),

    dict(student_id="EST_2018A_031", semestre=11, promedio_semestre=2.10, promedio_acumulado=2.75,
         tendencia_calificaciones=-0.30, creditos_matriculados=15, creditos_aprobados=6,
         porcentaje_avance_real=72.84, num_asignaturas_reprobadas=3, tasa_reprobacion=0.60,
         semestres_cursados=13, interrupciones_previas=2, anio_cohorte=2018, semestre_cohorte=1),

    # ── ALTO ─────────────────────────────────────────────────────────────────
    dict(student_id="EST_2021A_005", semestre=6,  promedio_semestre=2.54, promedio_acumulado=3.19,
         tendencia_calificaciones=-0.23, creditos_matriculados=21, creditos_aprobados=14,
         porcentaje_avance_real=41.98, num_asignaturas_reprobadas=2, tasa_reprobacion=0.43,
         semestres_cursados=6,  interrupciones_previas=0, anio_cohorte=2021, semestre_cohorte=2),

    dict(student_id="EST_2020B_018", semestre=7,  promedio_semestre=2.72, promedio_acumulado=3.05,
         tendencia_calificaciones=-0.15, creditos_matriculados=18, creditos_aprobados=11,
         porcentaje_avance_real=48.15, num_asignaturas_reprobadas=2, tasa_reprobacion=0.39,
         semestres_cursados=7,  interrupciones_previas=0, anio_cohorte=2020, semestre_cohorte=2),

    dict(student_id="EST_2021B_029", semestre=5,  promedio_semestre=2.40, promedio_acumulado=2.90,
         tendencia_calificaciones=-0.28, creditos_matriculados=18, creditos_aprobados=10,
         porcentaje_avance_real=32.10, num_asignaturas_reprobadas=2, tasa_reprobacion=0.44,
         semestres_cursados=6,  interrupciones_previas=1, anio_cohorte=2021, semestre_cohorte=2),

    dict(student_id="EST_2019A_044", semestre=8,  promedio_semestre=2.85, promedio_acumulado=3.22,
         tendencia_calificaciones=-0.08, creditos_matriculados=15, creditos_aprobados=9,
         porcentaje_avance_real=53.09, num_asignaturas_reprobadas=2, tasa_reprobacion=0.40,
         semestres_cursados=8,  interrupciones_previas=0, anio_cohorte=2019, semestre_cohorte=1),

    dict(student_id="EST_2022A_003", semestre=4,  promedio_semestre=2.30, promedio_acumulado=2.80,
         tendencia_calificaciones=-0.35, creditos_matriculados=21, creditos_aprobados=12,
         porcentaje_avance_real=22.22, num_asignaturas_reprobadas=3, tasa_reprobacion=0.57,
         semestres_cursados=5,  interrupciones_previas=1, anio_cohorte=2022, semestre_cohorte=1),

    # ── MODERADO ─────────────────────────────────────────────────────────────
    dict(student_id="EST_2022B_011", semestre=5,  promedio_semestre=3.72, promedio_acumulado=3.63,
         tendencia_calificaciones=0.09, creditos_matriculados=15, creditos_aprobados=12,
         porcentaje_avance_real=38.27, num_asignaturas_reprobadas=1, tasa_reprobacion=0.20,
         semestres_cursados=5,  interrupciones_previas=0, anio_cohorte=2022, semestre_cohorte=2),

    dict(student_id="EST_2021A_037", semestre=6,  promedio_semestre=3.50, promedio_acumulado=3.45,
         tendencia_calificaciones=0.05, creditos_matriculados=18, creditos_aprobados=14,
         porcentaje_avance_real=43.21, num_asignaturas_reprobadas=1, tasa_reprobacion=0.22,
         semestres_cursados=6,  interrupciones_previas=0, anio_cohorte=2021, semestre_cohorte=1),

    dict(student_id="EST_2022A_025", semestre=4,  promedio_semestre=3.30, promedio_acumulado=3.55,
         tendencia_calificaciones=-0.18, creditos_matriculados=18, creditos_aprobados=13,
         porcentaje_avance_real=27.78, num_asignaturas_reprobadas=2, tasa_reprobacion=0.28,
         semestres_cursados=4,  interrupciones_previas=0, anio_cohorte=2022, semestre_cohorte=1),

    dict(student_id="EST_2023B_008", semestre=3,  promedio_semestre=3.10, promedio_acumulado=3.20,
         tendencia_calificaciones=-0.25, creditos_matriculados=21, creditos_aprobados=15,
         porcentaje_avance_real=18.52, num_asignaturas_reprobadas=2, tasa_reprobacion=0.29,
         semestres_cursados=3,  interrupciones_previas=1, anio_cohorte=2023, semestre_cohorte=2),

    # ── BAJO ─────────────────────────────────────────────────────────────────
    dict(student_id="EST_2024A_002", semestre=2,  promedio_semestre=2.74, promedio_acumulado=3.02,
         tendencia_calificaciones=0.28, creditos_matriculados=15, creditos_aprobados=12,
         porcentaje_avance_real=12.96, num_asignaturas_reprobadas=1, tasa_reprobacion=0.20,
         semestres_cursados=2,  interrupciones_previas=0, anio_cohorte=2023, semestre_cohorte=2),

    dict(student_id="EST_2024B_009", semestre=1,  promedio_semestre=4.20, promedio_acumulado=4.20,
         tendencia_calificaciones=0.00, creditos_matriculados=18, creditos_aprobados=17,
         porcentaje_avance_real=9.26,  num_asignaturas_reprobadas=0, tasa_reprobacion=0.00,
         semestres_cursados=1,  interrupciones_previas=0, anio_cohorte=2024, semestre_cohorte=1),

    dict(student_id="EST_2024A_016", semestre=2,  promedio_semestre=3.85, promedio_acumulado=3.92,
         tendencia_calificaciones=0.07, creditos_matriculados=21, creditos_aprobados=20,
         porcentaje_avance_real=15.43, num_asignaturas_reprobadas=0, tasa_reprobacion=0.00,
         semestres_cursados=2,  interrupciones_previas=0, anio_cohorte=2024, semestre_cohorte=1),

    dict(student_id="EST_2023A_041", semestre=3,  promedio_semestre=4.05, promedio_acumulado=4.10,
         tendencia_calificaciones=0.05, creditos_matriculados=18, creditos_aprobados=17,
         porcentaje_avance_real=20.99, num_asignaturas_reprobadas=0, tasa_reprobacion=0.00,
         semestres_cursados=3,  interrupciones_previas=0, anio_cohorte=2023, semestre_cohorte=1),

    dict(student_id="EST_2023B_033", semestre=4,  promedio_semestre=3.60, promedio_acumulado=3.75,
         tendencia_calificaciones=-0.05, creditos_matriculados=21, creditos_aprobados=19,
         porcentaje_avance_real=27.16, num_asignaturas_reprobadas=0, tasa_reprobacion=0.00,
         semestres_cursados=4,  interrupciones_previas=0, anio_cohorte=2023, semestre_cohorte=2),

    dict(student_id="EST_2022B_047", semestre=5,  promedio_semestre=4.10, promedio_acumulado=4.00,
         tendencia_calificaciones=0.10, creditos_matriculados=15, creditos_aprobados=15,
         porcentaje_avance_real=35.80, num_asignaturas_reprobadas=0, tasa_reprobacion=0.00,
         semestres_cursados=5,  interrupciones_previas=0, anio_cohorte=2022, semestre_cohorte=2),
]


def calcular_derivados(p: dict) -> dict:
    ratio  = p["creditos_aprobados"] / max(p["creditos_matriculados"], 1)
    deficit = float(p["creditos_matriculados"] - p["creditos_aprobados"])
    f_nota  = (5.0 - p["promedio_acumulado"]) / 4.0
    f_repro = p["tasa_reprobacion"]
    f_avance = 1.0 - p["porcentaje_avance_real"] / 100.0
    f_interr = min(p["interrupciones_previas"] / 2.0, 1.0)
    score   = 0.40 * f_nota + 0.30 * f_repro + 0.20 * f_avance + 0.10 * f_interr
    return {**p, "ratio_aprobacion": ratio, "deficit_creditos": deficit, "score_riesgo_bruto": score}


def nivel_riesgo(proba: float, umbral: float) -> str:
    if proba < 0.20:    return "bajo"
    if proba < umbral:  return "moderado"
    if proba < 0.70:    return "alto"
    return "muy_alto"


def main():
    print("Cargando modelo...")
    pipeline, raw_meta = mejor_modelo(_MODELS)
    scaler   = pipeline.named_steps["scaler"]
    fn = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None

    umbral  = raw_meta.get("umbral", 0.5)
    nombre  = raw_meta.get("nombre", "SVM")
    version = f"v{raw_meta.get('version', '?')}"
    print(f"Modelo: {nombre} {version} — umbral {umbral:.4f}")

    # Orden canónico de features (igual que utils.py)
    if fn is None:
        fn = [
            "semestre", "promedio_semestre", "promedio_acumulado",
            "tendencia_calificaciones", "creditos_matriculados", "creditos_aprobados",
            "porcentaje_avance_real", "num_asignaturas_reprobadas", "tasa_reprobacion",
            "semestres_cursados", "interrupciones_previas",
            "anio_cohorte", "semestre_cohorte",
            "ratio_aprobacion", "deficit_creditos", "score_riesgo_bruto",
            "nota_x_avance", "repro_x_deficit", "flag_nota_critica", "flag_reprobacion",
            "flag_rezago", "flag_tendencia_neg", "flag_interrupcion",
            "semestre_sq", "n_factores_riesgo",
        ]

    init_db()
    Session = sessionmaker(bind=engine)

    ahora = datetime.now()
    resultados = []

    for i, perfil in enumerate(PERFILES):
        datos = calcular_derivados(perfil)
        student_id = datos.pop("student_id")
        semestre   = datos["semestre"]

        # Construir fila de features
        row = {**datos, "student_id": "seed", "deserta": 0}
        df  = pd.DataFrame([row])
        df  = agregar_features_avanzadas(df)
        X   = df[fn].values.astype(float)

        proba = float(pipeline.predict_proba(X)[0, 1])
        nivel = nivel_riesgo(proba, umbral)

        # Timestamp escalonado: últimos 30 días
        ts = ahora - timedelta(days=30 - i * 1.5)

        db = Session()
        try:
            reg = db.query(__import__("sapde.db.models", fromlist=["Prediccion"]).Prediccion).filter_by(student_id=student_id).first()
            if reg:
                db.delete(reg)
                db.commit()

            import json
            from sapde.db.models import Prediccion
            registro = Prediccion(
                student_id=student_id,
                semestre=semestre,
                probabilidad=round(proba, 6),
                prediccion=proba >= umbral,
                nivel_riesgo=nivel,
                umbral=umbral,
                modelo_nombre=nombre,
                modelo_version=version,
                features_json=json.dumps({k: (float(v) if hasattr(v, "item") else v) for k, v in datos.items()}),
                shap_json=None,
                timestamp=ts,
            )
            db.add(registro)
            db.commit()
        finally:
            db.close()

        resultados.append((student_id, semestre, nivel, proba))
        print(f"  {student_id:20s} sem={semestre:2d}  {nivel:10s}  {proba:.1%}")

    print(f"\nInsertados {len(resultados)} registros.")
    por_nivel = {}
    for _, _, nivel, _ in resultados:
        por_nivel[nivel] = por_nivel.get(nivel, 0) + 1
    for n, c in sorted(por_nivel.items()):
        print(f"  {n:10s}: {c}")


if __name__ == "__main__":
    main()
