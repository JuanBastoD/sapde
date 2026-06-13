"""
Módulo SHAP para interpretabilidad de modelos SAPDE (OE4).

Estrategia por tipo de modelo:
  - RandomForest, LightGBM → shap.TreeExplainer (rápido, milisegundos)
  - SVM (CalibratedClassifierCV) → shap.KernelExplainer con pipeline completo
  - MLP, ANFIS → shap.KernelExplainer con pipeline completo

Para modelos de árbol el input del explainer es X escalado (salida del
StandardScaler del pipeline). Para KernelExplainer se usa X sin escalar
y el pipeline.predict_proba() maneja internamente el escalado.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend no-interactivo para servidores / tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger("sapde.explain.shap")

# Modelos que usan TreeExplainer (por su nombre)
_TREE_MODELOS = {"RandomForest", "LightGBM"}


def _extraer_clase1(raw) -> np.ndarray:
    """
    Normaliza el output de TreeExplainer a shape (n, n_features).

    TreeExplainer puede devolver:
      - lista [class0_arr, class1_arr]: tomamos class1_arr
      - ndarray 3D (n, features, classes): tomamos [:, :, 1]
      - ndarray 2D (n, features): ya correcto
    """
    if isinstance(raw, list):
        return raw[1]
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _expected_value_clase1(ev) -> float:
    """Extrae el expected value para clase 1."""
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1])
    return float(ev)

# Número de muestras de fondo para KernelExplainer (k-means clustering)
_N_KMEANS = 50
# Número de muestras de evaluación por defecto para shap_values()
_N_EVAL_KERNEL = 150


def _es_arbol(nombre: str) -> bool:
    return nombre in _TREE_MODELOS


class ExplicadorSHAP:
    """
    Wrapper SHAP alrededor de un pipeline entrenado de SAPDE.

    Parámetros
    ----------
    nombre : str
        Nombre del modelo ("SVM", "RandomForest", …).
    pipeline : ImbPipeline
        Pipeline completo entrenado (SMOTE → Scaler → Modelo).
    feature_names : list[str]
        Nombres de las 25 features de entrada.
    X_background : np.ndarray
        Subconjunto del conjunto de entrenamiento/test para el background
        del explainer (sin escalar).
    n_background : int
        Máximo de muestras de fondo a usar.
    """

    def __init__(
        self,
        nombre: str,
        pipeline,
        feature_names: list[str],
        X_background: np.ndarray,
        n_background: int = 200,
    ):
        self.nombre = nombre
        self.pipeline = pipeline
        self.feature_names = feature_names

        idx = np.random.default_rng(42).choice(
            len(X_background),
            size=min(n_background, len(X_background)),
            replace=False,
        )
        self._X_bg_raw = X_background[idx]

        self._scaler = pipeline.named_steps["scaler"]
        self._modelo = pipeline.named_steps["modelo"]
        self._X_bg_scaled = self._scaler.transform(self._X_bg_raw)

        self._explainer = None    # inicialización diferida
        self._tipo: Optional[str] = None

    # ── Construcción del explainer ────────────────────────────────────────────

    def _inicializar_explainer(self) -> None:
        """Inicializa el explainer SHAP según el tipo de modelo."""
        if self._explainer is not None:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _es_arbol(self.nombre):
                self._explainer = shap.TreeExplainer(
                    self._modelo,
                    data=self._X_bg_scaled,
                    feature_perturbation="interventional",
                )
                self._tipo = "tree"
                logger.info("TreeExplainer inicializado para %s", self.nombre)

            else:
                # KernelExplainer: usa el pipeline completo con X sin escalar
                background_km = shap.kmeans(
                    self._X_bg_raw,
                    min(_N_KMEANS, len(self._X_bg_raw)),
                )

                def _predict_fn(X: np.ndarray) -> np.ndarray:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return self.pipeline.predict_proba(X)[:, 1]

                self._explainer = shap.KernelExplainer(_predict_fn, background_km)
                self._tipo = "kernel"
                logger.info("KernelExplainer inicializado para %s", self.nombre)

    # ── Cálculo de SHAP values ────────────────────────────────────────────────

    def calcular_shap(
        self,
        X_raw: np.ndarray,
        n_muestras: Optional[int] = None,
        nsamples_kernel: int = 100,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Calcula SHAP values para un conjunto de muestras.

        Retorna
        -------
        (shap_vals, indices)
            shap_vals : (n, n_features) — valor SHAP por feature por muestra
            indices   : índices usados si se muestreó, None si se usaron todos
        """
        self._inicializar_explainer()

        # Submuestra para KernelExplainer (lento O(n²))
        if self._tipo == "kernel" and n_muestras is not None and len(X_raw) > n_muestras:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_raw), size=n_muestras, replace=False)
            X_eval_raw = X_raw[idx]
        else:
            idx = None
            X_eval_raw = X_raw

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self._tipo == "tree":
                X_eval_scaled = self._scaler.transform(X_eval_raw)
                raw = self._explainer.shap_values(X_eval_scaled)
                shap_vals = _extraer_clase1(raw)
            else:
                shap_vals = self._explainer.shap_values(
                    X_eval_raw, nsamples=nsamples_kernel, silent=True
                )

        return shap_vals, idx

    # ── Importancia global ────────────────────────────────────────────────────

    def importancia_global(
        self,
        X_raw: np.ndarray,
        n_muestras: int = 300,
    ) -> pd.DataFrame:
        """
        Retorna DataFrame con importancia media |SHAP| por feature.

        Columnas: feature, importancia_media, importancia_std
        Ordenado descendente por importancia_media.
        """
        shap_vals, _ = self.calcular_shap(X_raw, n_muestras=n_muestras)

        df = pd.DataFrame({
            "feature":           self.feature_names,
            "importancia_media": np.abs(shap_vals).mean(axis=0),
            "importancia_std":   np.abs(shap_vals).std(axis=0),
        }).sort_values("importancia_media", ascending=False).reset_index(drop=True)

        return df

    # ── Explicación por estudiante ────────────────────────────────────────────

    def explicar_estudiante(
        self,
        x_raw: np.ndarray,
        nsamples_kernel: int = 200,
    ) -> dict:
        """
        Calcula la explicación SHAP para un único estudiante.

        Parámetros
        ----------
        x_raw : array 1-D de shape (n_features,)
            Vector de features sin escalar del estudiante.

        Retorna
        -------
        dict con claves: feature_names, shap_values, base_value, prediccion_proba
        """
        self._inicializar_explainer()
        X_single = x_raw.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self._tipo == "tree":
                X_sc = self._scaler.transform(X_single)
                raw = self._explainer.shap_values(X_sc)
                sv = _extraer_clase1(raw)[0]   # shape (n_features,)
                base_val = _expected_value_clase1(self._explainer.expected_value)
            else:
                sv = self._explainer.shap_values(
                    X_single, nsamples=nsamples_kernel, silent=True
                )[0]
                base_val = float(self._explainer.expected_value)

        proba = float(self.pipeline.predict_proba(X_single)[0, 1])

        return {
            "feature_names": self.feature_names,
            "shap_values":   sv.tolist(),
            "base_value":    float(base_val),
            "prediccion_proba": proba,
            "por_feature": dict(zip(self.feature_names, sv.tolist())),
        }

    # ── Visualizaciones ───────────────────────────────────────────────────────

    def plot_summary(
        self,
        X_raw: np.ndarray,
        ruta_salida: Path,
        n_muestras: int = 300,
        plot_type: str = "dot",
    ) -> None:
        """
        Genera el SHAP summary plot (beeswarm/dot) y lo guarda en disco.

        plot_type : "dot" (beeswarm) o "bar" (importancia media)
        """
        shap_vals, idx = self.calcular_shap(X_raw, n_muestras=n_muestras)
        X_disp = X_raw[idx] if idx is not None else X_raw

        # Para el beeswarm necesitamos X escalado para los colores
        if self._tipo == "tree":
            X_color = self._scaler.transform(X_disp)
        else:
            X_color = X_disp

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals,
            X_color,
            feature_names=self.feature_names,
            plot_type=plot_type,
            show=False,
            max_display=15,
        )
        plt.title(f"SHAP — {self.nombre}", fontsize=13, pad=10)
        plt.tight_layout()
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=120, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP summary guardado: %s", ruta_salida)

    def plot_bar(
        self,
        X_raw: np.ndarray,
        ruta_salida: Path,
        n_muestras: int = 300,
    ) -> None:
        """Genera el SHAP bar plot (importancia media) y lo guarda."""
        self.plot_summary(X_raw, ruta_salida, n_muestras=n_muestras, plot_type="bar")

    def plot_waterfall(
        self,
        x_raw: np.ndarray,
        ruta_salida: Path,
        label_estudiante: str = "Estudiante",
        nsamples_kernel: int = 200,
    ) -> None:
        """
        Genera el SHAP waterfall plot para un único estudiante.

        x_raw : array 1-D (n_features,)
        """
        explicacion = self.explicar_estudiante(x_raw, nsamples_kernel=nsamples_kernel)
        sv = np.array(explicacion["shap_values"])
        base_val = explicacion["base_value"]
        proba = explicacion["prediccion_proba"]

        exp_obj = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=x_raw,
            feature_names=self.feature_names,
        )

        fig, _ = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(exp_obj, max_display=15, show=False)
        plt.title(
            f"SHAP — {self.nombre} | {label_estudiante} | P(deserción)={proba:.3f}",
            fontsize=11,
            pad=10,
        )
        plt.tight_layout()
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=120, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP waterfall guardado: %s", ruta_salida)

    def plot_importancia_comparativa(
        self,
        df_importancia: pd.DataFrame,
        ruta_salida: Path,
        top_n: int = 15,
    ) -> None:
        """Barras horizontales con la importancia media |SHAP| de las top features."""
        df_top = df_importancia.head(top_n)
        fig, ax = plt.subplots(figsize=(9, 6))
        colores = ["#e74c3c" if v > 0 else "#3498db" for v in df_top["importancia_media"]]
        ax.barh(df_top["feature"][::-1], df_top["importancia_media"][::-1],
                color=colores[::-1], edgecolor="white", height=0.7)
        ax.set_xlabel("Importancia media |SHAP|", fontsize=11)
        ax.set_title(f"Top {top_n} features — {self.nombre}", fontsize=13)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ruta_salida, dpi=120, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP importancia guardada: %s", ruta_salida)
