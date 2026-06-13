"""
ANFIS — Sistema Neuro-Difuso Adaptativo (OE2)

Implementación tipo Takagi-Sugeno de primer orden compatible con sklearn.

Arquitectura de 5 capas:
  Capa 1 — Funciones de membresía Gaussianas: mu_rk(x_k) = exp(-(x-c)²/2σ²)
  Capa 2 — Fuerza de disparo: w_r = prod_k(mu_rk)  (T-norma producto)
  Capa 3 — Fuerza normalizada: w̄_r = w_r / sum_r(w_r)
  Capa 4 — Consecuente lineal: f_r = a_r0 + sum_k(a_rk * x_k)
  Capa 5 — Salida: sigmoid(sum_r(w̄_r * f_r))

Entrenamiento: L-BFGS-B sobre binary cross-entropy (scipy.optimize.minimize).
Inicialización de centros: K-Means sobre el dataset de entrenamiento.
"""

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid estable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger("sapde.models.anfis")

_EPSILON = 1e-8  # evitar división por cero y log(0)


class ANFISClassifier(BaseEstimator, ClassifierMixin):
    """
    ANFIS Takagi-Sugeno para clasificación binaria de deserción estudiantil.

    Parámetros
    ----------
    n_reglas : int
        Número de reglas difusas (clusters en el espacio de entrada).
    n_componentes_pca : int o None
        Si no es None, aplica PCA antes del ANFIS para reducir dimensionalidad.
        Recomendado cuando n_features > 15.
    max_iter : int
        Iteraciones máximas del optimizador L-BFGS-B.
    lambda_reg : float
        Coeficiente de regularización L2 para los parámetros.
    tol : float
        Tolerancia de convergencia del optimizador.
    random_state : int
        Semilla para reproducibilidad.
    """

    def __init__(
        self,
        n_reglas: int = 6,
        n_componentes_pca: Optional[int] = 8,
        max_iter: int = 150,
        lambda_reg: float = 1e-3,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.n_reglas = n_reglas
        self.n_componentes_pca = n_componentes_pca
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.random_state = random_state

    # ── Helpers internos ───────────────────────────────────────────────────────

    def _reducir_dimensiones(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Aplica PCA si está configurado."""
        if self.n_componentes_pca is None or X.shape[1] <= self.n_componentes_pca:
            return X
        if fit:
            self._pca = PCA(n_components=self.n_componentes_pca,
                            random_state=self.random_state)
            return self._pca.fit_transform(X)
        return self._pca.transform(X)

    def _pack(self, centros, sigmas, consecuentes) -> np.ndarray:
        return np.concatenate([centros.ravel(), sigmas.ravel(), consecuentes.ravel()])

    def _unpack(self, theta: np.ndarray, n: int):
        r = self.n_reglas
        centros      = theta[:r * n].reshape(r, n)
        sigmas       = theta[r * n: 2 * r * n].reshape(r, n)
        consecuentes = theta[2 * r * n:].reshape(r, n + 1)
        return centros, sigmas, consecuentes

    def _memberships(self, X: np.ndarray, centros, sigmas) -> np.ndarray:
        """
        Fuerza de disparo de cada regla.

        X: (N, K)  →  retorna W: (N, R)
        """
        # diff[i, r, k] = x[i,k] - c[r,k]
        diff = X[:, np.newaxis, :] - centros[np.newaxis, :, :]   # (N, R, K)
        sig2 = (np.abs(sigmas) + _EPSILON)[np.newaxis, :, :]     # (1, R, K)
        mu   = np.exp(-0.5 * (diff / sig2) ** 2)                 # (N, R, K)
        W    = mu.prod(axis=2)                                    # (N, R) — T-norma producto
        return W

    def _forward(self, X: np.ndarray, centros, sigmas, consecuentes):
        """Pase adelante: devuelve probabilidad y fuerzas normalizadas."""
        N = X.shape[0]
        W       = self._memberships(X, centros, sigmas)           # (N, R)
        W_sum   = W.sum(axis=1, keepdims=True) + _EPSILON
        W_norm  = W / W_sum                                       # (N, R)

        X_aug   = np.hstack([np.ones((N, 1)), X])                # (N, K+1)
        F       = X_aug @ consecuentes.T                          # (N, R)

        output  = (W_norm * F).sum(axis=1)                       # (N,)
        proba   = expit(output)                                   # (N,) via sigmoid
        return proba, W_norm, F

    def _loss_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Binary cross-entropy + L2. Usa diferenciación automática numérica."""
        n = X.shape[1]
        centros, sigmas, consecuentes = self._unpack(theta, n)
        sigmas_pos = np.abs(sigmas) + _EPSILON

        proba, W_norm, F = self._forward(X, centros, sigmas_pos, consecuentes)
        proba = np.clip(proba, _EPSILON, 1.0 - _EPSILON)

        # BCE + L2
        bce = -np.mean(y * np.log(proba) + (1.0 - y) * np.log(1.0 - proba))
        l2  = self.lambda_reg * np.sum(theta ** 2)
        return bce + l2

    # ── API sklearn ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANFISClassifier":
        X, y = check_X_y(X, y)
        y    = y.astype(float)

        # Reducción opcional de dimensiones
        Xr = self._reducir_dimensiones(X, fit=True)
        n  = Xr.shape[1]

        logger.info(
            "ANFIS fit: %d muestras, %d features (%d reglas, max_iter=%d)",
            len(y), n, self.n_reglas, self.max_iter,
        )

        # ── Inicialización con K-Means ──────────────────────────────────────
        km = KMeans(n_clusters=self.n_reglas, random_state=self.random_state,
                    n_init="auto")
        km.fit(Xr)
        centros = km.cluster_centers_                             # (R, K)

        # Sigmas: desviación estándar dentro de cada cluster
        sigmas = np.ones((self.n_reglas, n)) * 0.5
        for r in range(self.n_reglas):
            mask = km.labels_ == r
            if mask.sum() > 1:
                sigmas[r] = Xr[mask].std(axis=0).clip(0.1) + _EPSILON

        rng = np.random.RandomState(self.random_state)
        consecuentes = rng.randn(self.n_reglas, n + 1) * 0.05

        theta0 = self._pack(centros, sigmas, consecuentes)

        # ── Optimización L-BFGS-B ───────────────────────────────────────────
        resultado = minimize(
            self._loss_grad,
            theta0,
            args=(Xr, y),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": 1e-5},
        )

        self.centros_, sigmas_opt, self.consecuentes_ = self._unpack(resultado.x, n)
        self.sigmas_  = np.abs(sigmas_opt) + _EPSILON
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self._loss_final = resultado.fun
        self._converged  = resultado.success

        logger.info(
            "ANFIS convergencia=%s  loss_final=%.4f",
            self._converged, self._loss_final,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["centros_", "sigmas_", "consecuentes_"])
        X = check_array(X)
        Xr = self._reducir_dimensiones(X, fit=False)
        proba, _, _ = self._forward(Xr, self.centros_, self.sigmas_, self.consecuentes_)
        proba = np.clip(proba, 0.0, 1.0)
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_reglas_activas(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve la regla de mayor fuerza de disparo para cada muestra.
        Útil para interpretabilidad: qué 'patrón difuso' activó la predicción.
        """
        check_is_fitted(self)
        X  = check_array(X)
        Xr = self._reducir_dimensiones(X, fit=False)
        W  = self._memberships(Xr, self.centros_, self.sigmas_)
        return W.argmax(axis=1)
