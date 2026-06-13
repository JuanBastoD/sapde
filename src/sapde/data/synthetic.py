"""
Generador de dataset sintético de deserción estudiantil.

Produce registros estudiante-semestre con correlaciones realistas entre
rendimiento académico, avance curricular y probabilidad de deserción.

Escala de calificaciones: 0.0 – 5.0 (sistema colombiano).
Nota de aprobación: 3.0.

# TODO: reemplazar por datos institucionales reales
# Punto de integración: función `cargar_datos_reales()` al final del módulo.
# El CSV real debe tener las mismas columnas que genera este script.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("sapde.data.synthetic")

# --- Constantes del programa de Ingeniería de Sistemas ---
CREDITOS_TOTALES_PROGRAMA = 162   # créditos para graduarse
ASIGNATURAS_POR_SEMESTRE_MIN = 4
ASIGNATURAS_POR_SEMESTRE_MAX = 7
CREDITOS_POR_ASIGNATURA = 3       # promedio


def _probabilidad_desercion(riesgo: np.ndarray) -> np.ndarray:
    """
    Convierte el factor latente de riesgo en probabilidad de deserción.
    Usa función logística centrada para obtener ~20-30% de positivos.
    """
    return 1.0 / (1.0 + np.exp(-(riesgo * 7.0 - 2.8)))


def _calcular_tendencia(notas: list[float]) -> float:
    """Pendiente de regresión lineal sobre notas históricas (OLS)."""
    n = len(notas)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), np.mean(notas)
    num = np.sum((x - xm) * (np.array(notas) - ym))
    den = np.sum((x - xm) ** 2)
    return float(num / den) if den != 0 else 0.0


def generar_dataset(
    n_estudiantes: int = 1500,
    semilla: int = 42,
    tasa_desercion_objetivo: float = 0.25,
) -> pd.DataFrame:
    """
    Genera el dataset sintético estudiante-semestre.

    Parámetros
    ----------
    n_estudiantes : int
        Número de estudiantes únicos a simular.
    semilla : int
        Semilla fija para reproducibilidad.
    tasa_desercion_objetivo : float
        Fracción aproximada de estudiantes desertores (entre 0.20 y 0.30).

    Retorna
    -------
    pd.DataFrame con ~5 000–10 000 filas y las columnas del esquema SAPDE.
    """
    rng = np.random.default_rng(semilla)
    logger.info(
        "Generando dataset sintético: %d estudiantes, semilla=%d", n_estudiantes, semilla
    )

    # --- Factor latente de riesgo por estudiante ---
    # Beta(2, 6): la mayoría tiene bajo riesgo, cola larga hacia alto riesgo
    riesgo_base = rng.beta(2, 6, size=n_estudiantes)

    # --- Cohortes disponibles (últimos 5 años) ---
    cohortes = [
        f"{año}-{sem}"
        for año in range(2019, 2024)
        for sem in ["I", "II"]
    ]

    registros: list[dict] = []

    for i in range(n_estudiantes):
        riesgo = riesgo_base[i]
        cohorte = rng.choice(cohortes)

        # Número máximo de semestres cursados:
        # Estudiantes de alto riesgo tienden a abandonar pronto
        if riesgo > 0.65:
            max_sem = int(rng.integers(1, 5))
        elif riesgo > 0.40:
            max_sem = int(rng.integers(2, 8))
        else:
            max_sem = int(rng.integers(5, 11))

        # ¿Este estudiante deserta?
        prob_desertar = _probabilidad_desercion(np.array([riesgo]))[0]
        deserta_final = bool(rng.random() < prob_desertar)

        # Interrupciones académicas (semestres sin matricularse)
        interrupciones = 0
        if riesgo > 0.45 and rng.random() < 0.30:
            interrupciones = int(rng.integers(1, 3))

        # Acumuladores por semestre
        notas_hist: list[float] = []
        creditos_acumulados = 0

        for sem in range(1, max_sem + 1):

            # --- Calificación semestral (escala 0-5, aprobación ≥ 3.0) ---
            # Alto riesgo → notas más bajas y mayor varianza
            media_nota = 4.3 - 2.8 * riesgo
            std_nota = 0.35 + 0.30 * riesgo
            promedio_sem = float(
                np.clip(rng.normal(media_nota, std_nota), 1.0, 5.0)
            )
            notas_hist.append(promedio_sem)

            # --- Créditos y asignaturas ---
            n_asignaturas = int(rng.integers(
                ASIGNATURAS_POR_SEMESTRE_MIN,
                ASIGNATURAS_POR_SEMESTRE_MAX + 1
            ))
            creditos_mat = n_asignaturas * CREDITOS_POR_ASIGNATURA

            # Tasa de aprobación correlacionada con riesgo
            tasa_aprobacion = float(
                np.clip(rng.normal(0.95 - 0.85 * riesgo, 0.08), 0.0, 1.0)
            )
            creditos_aprobados = int(round(creditos_mat * tasa_aprobacion))
            creditos_acumulados += creditos_aprobados

            # Asignaturas reprobadas (Poisson con λ basado en riesgo)
            lam_reprobadas = max(0.0, riesgo * 3.5)
            num_reprobadas = int(min(rng.poisson(lam_reprobadas), n_asignaturas))

            # Tasa de reprobación (sobre asignaturas del semestre)
            tasa_reprobacion = round(num_reprobadas / max(1, n_asignaturas), 4)

            # --- Métricas acumuladas ---
            promedio_acumulado = float(np.mean(notas_hist))
            tendencia = _calcular_tendencia(notas_hist)
            porcentaje_avance = min(
                100.0,
                creditos_acumulados / CREDITOS_TOTALES_PROGRAMA * 100.0
            )

            # --- Etiqueta de deserción ---
            # Solo el último semestre de un desertor lleva deserta=1
            es_ultimo_sem = (sem == max_sem)
            deserta_sem = int(deserta_final and es_ultimo_sem)

            registros.append({
                "student_id": f"EST{i:05d}",
                "cohorte": cohorte,
                "semestre": sem,
                # Rendimiento
                "promedio_semestre": round(promedio_sem, 2),
                "promedio_acumulado": round(promedio_acumulado, 2),
                "tendencia_calificaciones": round(tendencia, 4),
                # Avance curricular
                "creditos_matriculados": creditos_mat,
                "creditos_aprobados": creditos_aprobados,
                "porcentaje_avance_real": round(porcentaje_avance, 2),
                # Reprobación
                "num_asignaturas_reprobadas": num_reprobadas,
                "tasa_reprobacion": tasa_reprobacion,
                # Permanencia
                "semestres_cursados": sem,
                "interrupciones_previas": interrupciones,
                # Variable objetivo
                "deserta": deserta_sem,
            })

    df = pd.DataFrame(registros)

    # Reordenar columnas en el orden canónico del esquema SAPDE
    columnas_orden = [
        "student_id", "cohorte", "semestre",
        "promedio_semestre", "promedio_acumulado", "tendencia_calificaciones",
        "creditos_matriculados", "creditos_aprobados", "porcentaje_avance_real",
        "num_asignaturas_reprobadas", "tasa_reprobacion",
        "semestres_cursados", "interrupciones_previas",
        "deserta",
    ]
    df = df[columnas_orden]

    # --- Estadísticas del dataset generado ---
    n_desertores = df.groupby("student_id")["deserta"].max().sum()
    tasa_real = n_desertores / n_estudiantes
    logger.info(
        "Dataset generado: %d registros, %d estudiantes únicos, "
        "%d desertores (%.1f%%)",
        len(df), n_estudiantes, n_desertores, tasa_real * 100,
    )

    if not (0.15 <= tasa_real <= 0.40):
        logger.warning(
            "Tasa de deserción %.1f%% fuera del rango esperado (15%%–40%%). "
            "Verifica los parámetros del generador.",
            tasa_real * 100,
        )

    return df


def guardar_dataset(df: pd.DataFrame, ruta_csv: Path) -> None:
    """Guarda el dataset en formato CSV con metadatos en el nombre de archivo."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    logger.info("Dataset guardado en: %s (%d filas)", ruta_csv, len(df))


# TODO: reemplazar por datos institucionales reales
def cargar_datos_reales(ruta: Path) -> pd.DataFrame:
    """
    Punto de integración con datos institucionales.

    El CSV real debe contener exactamente las mismas columnas que produce
    `generar_dataset()`. Esta función valida el esquema antes de devolver
    el DataFrame.

    Parámetros
    ----------
    ruta : Path
        Ruta al CSV con los datos reales de la Universidad de Pamplona.
    """
    columnas_requeridas = {
        "student_id", "cohorte", "semestre",
        "promedio_semestre", "promedio_acumulado", "tendencia_calificaciones",
        "creditos_matriculados", "creditos_aprobados", "porcentaje_avance_real",
        "num_asignaturas_reprobadas", "tasa_reprobacion",
        "semestres_cursados", "interrupciones_previas",
        "deserta",
    }

    df = pd.read_csv(ruta, encoding="utf-8")
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes en el CSV institucional: {faltantes}")

    logger.info("Datos reales cargados desde %s (%d filas)", ruta, len(df))
    return df


def main() -> None:
    """Punto de entrada CLI: genera y guarda el dataset sintético."""
    from sapde.config import obtener_config, configurar_logging

    configurar_logging()
    cfg = obtener_config()
    rutas = cfg.rutas_absolutas()

    ruta_csv = rutas["raw"] / "estudiantes_sintetico.csv"
    df = generar_dataset(
        n_estudiantes=cfg.synthetic_n_estudiantes,
        semilla=cfg.synthetic_semilla,
    )
    guardar_dataset(df, ruta_csv)
    print(f"\nDataset guardado en: {ruta_csv}")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
