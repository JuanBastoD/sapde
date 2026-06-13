"""
01_eda.py — Análisis Exploratorio de Datos (OE1: Análisis de patrones de deserción)

Ejecutar:
    cd sapde
    python notebooks/01_eda.py

Genera:
    reports/eda_distribucion_variables.png
    reports/eda_boxplots_riesgo.png
    reports/eda_correlacion.png
    reports/eda_tasa_desercion_cohorte.png
    reports/eda_score_riesgo.png
    reports/eda_resumen.txt
"""

import sys
import logging
from pathlib import Path

# Agregar src al path para importar sapde
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")  # backend sin ventana para guardar PNGs

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("sapde.eda")

# Paleta de colores del proyecto
COLOR_PERMANECE = "#2196F3"   # azul
COLOR_DESERTOR  = "#F44336"   # rojo
PALETTE         = [COLOR_PERMANECE, COLOR_DESERTOR]

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_datos(raiz: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el dataset procesado (si existe) o el raw.
    Retorna (df_fila, df_estudiante) donde df_estudiante agrega al nivel persona.
    """
    etl_csv = raiz / "data" / "interim" / "estudiantes_etl.csv"
    raw_csv = raiz / "data" / "raw" / "estudiantes_sintetico.csv"

    if etl_csv.exists():
        logger.info("Cargando datos ETL: %s", etl_csv)
        df = pd.read_csv(etl_csv, encoding="utf-8")
    elif raw_csv.exists():
        logger.info("ETL no ejecutado. Cargando raw: %s", raw_csv)
        df = pd.read_csv(raw_csv, encoding="utf-8")
    else:
        raise FileNotFoundError(
            "No se encontró dataset. Ejecuta: python -m sapde.data.synthetic"
        )

    # Vista a nivel estudiante: tomar el último semestre de cada uno
    df_est = (
        df.sort_values("semestre")
        .groupby("student_id")
        .last()
        .reset_index()
    )
    # Variable objetivo a nivel estudiante
    df_est["deserta_final"] = df.groupby("student_id")["deserta"].max().values

    logger.info(
        "Dataset: %d filas, %d estudiantes, %d desertores (%.1f%%)",
        len(df), len(df_est),
        df_est["deserta_final"].sum(),
        df_est["deserta_final"].mean() * 100,
    )
    return df, df_est


# ── Figura 1: Distribución de variables clave ─────────────────────────────────

def plot_distribucion_variables(df: pd.DataFrame, ruta: Path) -> None:
    """Histogramas de las variables numéricas clave por grupo (permanece vs deserta)."""
    variables = [
        ("promedio_acumulado",         "Promedio acumulado (0-5)"),
        ("tasa_reprobacion",           "Tasa de reprobación"),
        ("porcentaje_avance_real",     "% Avance curricular"),
        ("tendencia_calificaciones",   "Tendencia de calificaciones"),
        ("semestres_cursados",         "Semestres cursados"),
        ("interrupciones_previas",     "Interrupciones previas"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Distribución de variables clave — SAPDE\n(azul = permanece, rojo = deserta)",
                 fontsize=13, fontweight="bold")

    for ax, (col, titulo) in zip(axes.flat, variables):
        for val, color, etiq in [(0, COLOR_PERMANECE, "Permanece"),
                                  (1, COLOR_DESERTOR,  "Deserta")]:
            datos = df.loc[df["deserta"] == val, col].dropna()
            ax.hist(datos, bins=25, alpha=0.55, color=color,
                    label=etiq, density=True)
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


# ── Figura 2: Boxplots por estatus de deserción ───────────────────────────────

def plot_boxplots_riesgo(df_est: pd.DataFrame, ruta: Path) -> None:
    """Boxplots de variables críticas comparando desertores vs permanentes."""
    variables = [
        ("promedio_acumulado",     "Promedio acumulado"),
        ("tasa_reprobacion",       "Tasa reprobación"),
        ("porcentaje_avance_real", "% Avance curricular"),
        ("semestres_cursados",     "Semestres cursados"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("Diferencias entre desertores y permanentes — nivel estudiante",
                 fontsize=12, fontweight="bold")

    df_plot = df_est.copy()
    df_plot["Estado"] = df_plot["deserta_final"].map({0: "Permanece", 1: "Deserta"})

    for ax, (col, titulo) in zip(axes, variables):
        sns.boxplot(
            data=df_plot, x="Estado", y=col, ax=ax,
            hue="Estado",
            palette={"Permanece": COLOR_PERMANECE, "Deserta": COLOR_DESERTOR},
            width=0.5, linewidth=1.2, legend=False,
        )
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


# ── Figura 3: Matriz de correlación ───────────────────────────────────────────

def plot_correlacion(df: pd.DataFrame, ruta: Path) -> None:
    """Mapa de calor con correlaciones de Pearson entre variables numéricas."""
    numericas = [
        "promedio_semestre", "promedio_acumulado", "tendencia_calificaciones",
        "tasa_reprobacion", "porcentaje_avance_real",
        "num_asignaturas_reprobadas", "creditos_aprobados",
        "semestres_cursados", "interrupciones_previas", "deserta",
    ]
    # Solo columnas que existen en el DataFrame
    numericas = [c for c in numericas if c in df.columns]
    corr = df[numericas].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Matriz de correlación — variables SAPDE", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


# ── Figura 4: Tasa de deserción por cohorte ───────────────────────────────────

def plot_desercion_por_cohorte(df_est: pd.DataFrame, ruta: Path) -> None:
    """Barras con la tasa de deserción por cohorte de ingreso."""
    tasa = (
        df_est.groupby("cohorte")["deserta_final"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "tasa_desercion", "count": "n_estudiantes"})
        .reset_index()
        .sort_values("cohorte")
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Deserción por cohorte de ingreso", fontsize=12, fontweight="bold")

    # Tasa de deserción
    colores = [COLOR_DESERTOR if t > 0.30 else COLOR_PERMANECE
               for t in tasa["tasa_desercion"]]
    bars = ax1.bar(tasa["cohorte"], tasa["tasa_desercion"] * 100,
                   color=colores, alpha=0.8, edgecolor="white")
    ax1.axhline(tasa["tasa_desercion"].mean() * 100, color="gray",
                linestyle="--", linewidth=1.5, label="Promedio global")
    ax1.set_ylabel("Tasa de deserción (%)")
    ax1.set_title("Tasa de deserción por cohorte")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Número de estudiantes por cohorte
    ax2.bar(tasa["cohorte"], tasa["n_estudiantes"], color="#90A4AE", alpha=0.8)
    ax2.set_ylabel("N° estudiantes")
    ax2.set_title("Estudiantes por cohorte")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


# ── Figura 5: Score de riesgo y separabilidad ────────────────────────────────

def plot_score_riesgo(df_est: pd.DataFrame, ruta: Path) -> None:
    """Distribución del score_riesgo_bruto por grupo (si existe) o promedio_acumulado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Separabilidad entre desertores y permanentes", fontsize=12, fontweight="bold")

    col_score = "score_riesgo_bruto" if "score_riesgo_bruto" in df_est.columns else "promedio_acumulado"
    titulo_score = "Score de riesgo bruto (ETL)" if col_score == "score_riesgo_bruto" \
                   else "Promedio acumulado (sin ETL)"

    # Histograma del score
    ax = axes[0]
    for val, color, etiq in [(0, COLOR_PERMANECE, "Permanece"),
                              (1, COLOR_DESERTOR,  "Deserta")]:
        datos = df_est.loc[df_est["deserta_final"] == val, col_score].dropna()
        ax.hist(datos, bins=30, alpha=0.6, color=color,
                label=f"{etiq} (n={len(datos)})", density=True)
    ax.set_xlabel(titulo_score)
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de {titulo_score}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Scatter: nota acumulada vs tasa de reprobación, coloreado por deserción
    ax2 = axes[1]
    scatter_cols = {
        0: df_est[df_est["deserta_final"] == 0],
        1: df_est[df_est["deserta_final"] == 1],
    }
    for val, df_g in scatter_cols.items():
        color = COLOR_PERMANECE if val == 0 else COLOR_DESERTOR
        etiq  = "Permanece" if val == 0 else "Deserta"
        ax2.scatter(
            df_g["promedio_acumulado"],
            df_g["tasa_reprobacion"],
            c=color, alpha=0.4, s=15, label=etiq,
        )
    ax2.set_xlabel("Promedio acumulado")
    ax2.set_ylabel("Tasa de reprobación")
    ax2.set_title("Rendimiento vs Reprobación")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    logger.info("Guardado: %s", ruta)


# ── Resumen textual ───────────────────────────────────────────────────────────

def generar_resumen(df: pd.DataFrame, df_est: pd.DataFrame, ruta: Path) -> None:
    """Genera un archivo de texto con las estadísticas descriptivas clave."""
    lineas = [
        "=" * 60,
        "SAPDE — Análisis Exploratorio de Datos (OE1)",
        "=" * 60,
        "",
        f"Total de registros (filas estudiante-semestre): {len(df):,}",
        f"Estudiantes únicos:                             {len(df_est):,}",
        f"Desertores (nivel estudiante):                  {df_est['deserta_final'].sum():,}",
        f"Tasa de deserción:                              {df_est['deserta_final'].mean():.1%}",
        "",
        "-- Estadisticas por grupo ------------------------------------------",
    ]

    for grupo, nombre in [(0, "PERMANENTES"), (1, "DESERTORES")]:
        sub = df_est[df_est["deserta_final"] == grupo]
        lineas += [
            f"\n{nombre} (n={len(sub)}):",
            f"  Promedio acumulado:    {sub['promedio_acumulado'].mean():.2f} +/- {sub['promedio_acumulado'].std():.2f}",
            f"  Tasa de reprobacion:   {sub['tasa_reprobacion'].mean():.3f} +/- {sub['tasa_reprobacion'].std():.3f}",
            f"  % Avance curricular:   {sub['porcentaje_avance_real'].mean():.1f}% +/- {sub['porcentaje_avance_real'].std():.1f}%",
            f"  Semestres cursados:    {sub['semestres_cursados'].mean():.1f} +/- {sub['semestres_cursados'].std():.1f}",
        ]

    lineas += [
        "",
        "-- Correlaciones con deserta ----------------------------------------",
    ]
    correlaciones = df.select_dtypes(include=[float, int]).corr()["deserta"].drop("deserta")
    correlaciones_ord = correlaciones.abs().sort_values(ascending=False)
    for col in correlaciones_ord.index[:8]:
        val = correlaciones[col]
        lineas.append(f"  {col:<35} r = {val:+.3f}")

    lineas += [
        "",
        "-- Factores de riesgo identificados ----------------------------------",
        "  1. Promedio acumulado bajo (< 3.2)  -> mayor riesgo",
        "  2. Alta tasa de reprobacion          -> mayor riesgo",
        "  3. Bajo avance curricular             -> mayor riesgo",
        "  4. Tendencia negativa de notas        -> mayor riesgo",
        "  5. Interrupciones academicas previas  -> mayor riesgo",
        "",
        "Graficas generadas en reports/eda_*.png",
        "=" * 60,
    ]

    texto = "\n".join(lineas)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    logger.info("Guardado: %s", ruta)
    # Imprimir con encoding seguro para consola Windows
    print(texto.encode("ascii", errors="replace").decode("ascii"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    raiz    = Path(__file__).resolve().parents[1]
    reports = raiz / "reports"
    reports.mkdir(exist_ok=True)

    logger.info("=== Análisis Exploratorio SAPDE (OE1) ===")
    df, df_est = cargar_datos(raiz)

    plot_distribucion_variables(df,     reports / "eda_distribucion_variables.png")
    plot_boxplots_riesgo(df_est,        reports / "eda_boxplots_riesgo.png")
    plot_correlacion(df,                reports / "eda_correlacion.png")
    plot_desercion_por_cohorte(df_est,  reports / "eda_tasa_desercion_cohorte.png")
    plot_score_riesgo(df_est,           reports / "eda_score_riesgo.png")
    generar_resumen(df, df_est,         reports / "eda_resumen.txt")

    logger.info("EDA completado. Revisa la carpeta reports/")


if __name__ == "__main__":
    main()
