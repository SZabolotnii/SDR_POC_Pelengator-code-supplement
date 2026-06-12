"""Kunchenko/DSGE feature generation.

The core mathematical object is the per-class reconstruction model:

    F_c * K_c = B_c                                         (system 4 in skill)

where
    F_c[i,j] = E_c[(phi_i(x) - psi_i)(phi_j(x) - psi_j)]   (S x S matrix)
    B_c[i]   = E_c[(x - psi_0)(phi_i(x) - psi_i)]          (S vector)
    psi_0    = E_c[x]
    psi_i    = E_c[phi_i(x)]

The classifier features are log-MSED reconstruction errors per class:

    eps_c(X_test) = log( sum_j ||x_j - x_hat_j^(c)||^2 + eps )

These features are small when X_test belongs to class c, and large
otherwise — effectively a class-conditional reconstruction distance.

Public API:
    KunchenkoReconstructor(basis, n, alpha=None, ridge=0.01)
        .fit_class(X_class) -> per-class K, k0, psi_0, psi
        .reconstruct(x) -> x_hat
        .mse_reconstruction(X) -> scalar MSE

    DSGEFeatureExtractor(basis, n, alpha=None, ridge=0.01, eps=1e-8)
        .fit(X, y) -> learn one reconstructor per class
        .transform(X) -> (n_samples, n_classes) log-MSED feature matrix
        .fit_transform(X, y)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basis_functions import TensorProductBasis, get_basis  # noqa: E402


class KunchenkoReconstructor:
    """Solve F * K = B per class with Tikhonov regularisation.

    `basis` is a string from {'polynomial', 'fractional', 'trigonometric',
    'robust', 'log', 'patp'}. For 'patp', supply alpha.

    `ridge` is the Tikhonov regularisation parameter; 0.01 is a sensible
    default for typical embedding/sensor scales. Increase if F is poorly
    conditioned (cond(F) > 1e6 is a warning sign).
    """

    def __init__(self, basis: str = "fractional", n: int = 3,
                 alpha: float | None = None, ridge: float = 0.01):
        self.basis_name = basis
        self.n = n
        self.alpha = alpha
        self.ridge = ridge
        self.phi = get_basis(basis, n=n, alpha=alpha)

        # Fitted parameters (None until fit_class is called)
        self.K: np.ndarray | None = None
        self.k0: float | None = None
        self.psi_0: float | None = None
        self.psi: np.ndarray | None = None
        self.cond_F: float | None = None

    def _basis_matrix(self, x_flat: np.ndarray) -> np.ndarray:
        """Component-wise basis: (N,) -> (N, n)."""
        return self.phi(x_flat)

    def fit_class(self, X_class: np.ndarray) -> "KunchenkoReconstructor":
        """Fit reconstructor on data from a single class.

        X_class: (n_samples, dim) array. The reconstructor operates
        component-wise, so all components are pooled to estimate moments.
        """
        x = X_class.ravel().astype(np.float64)
        Phi = self._basis_matrix(x)            # (N, n)

        self.psi_0 = float(np.mean(x))
        self.psi = np.mean(Phi, axis=0)        # (n,)

        Phi_c = Phi - self.psi                  # centred basis
        x_c = x - self.psi_0

        # F = E[(phi - psi)(phi - psi)^T], B = E[(x - psi_0)(phi - psi)]
        F = (Phi_c.T @ Phi_c) / len(x)
        B = (Phi_c.T @ x_c) / len(x)

        # Tikhonov regularisation for conditioning
        F_reg = F + self.ridge * np.eye(self.n)
        self.cond_F = float(np.linalg.cond(F_reg))

        self.K = np.linalg.solve(F_reg, B)
        self.k0 = self.psi_0 - float(self.K @ self.psi)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted reconstruction component-wise.

        Returns x_hat of same shape as X.
        """
        if self.K is None:
            raise RuntimeError("Reconstructor not fitted; call fit_class first.")
        original_shape = X.shape
        x = X.ravel().astype(np.float64)
        Phi = self._basis_matrix(x)
        x_hat = self.k0 + Phi @ self.K
        return x_hat.reshape(original_shape)

    def mse_per_sample(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE between X and its reconstruction.

        For X of shape (n_samples, dim), returns array of shape (n_samples,)
        with the mean of squared component-wise reconstruction errors.
        """
        X = np.atleast_2d(X.astype(np.float64))
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)


class DSGEFeatureExtractor:
    """Per-class reconstruction with log-MSED features.

    Workflow:
        fit(X, y)        -> trains one KunchenkoReconstructor per class
        transform(X)     -> (n_samples, n_classes) log-MSED matrix
                            (small value at column c => sample is class c)

    Use the resulting feature matrix as input to any classifier
    (LogisticRegression, SVM, RandomForest, ...). For best results:
      - normalise X first (StandardScaler);
      - tune `basis` and `n` on a held-out validation set;
      - for hybrid models, concatenate the (n_samples, n_classes) DSGE
        features with traditional features before classification.
    """

    def __init__(self, basis: str = "fractional", n: int = 3,
                 alpha: float | None = None, ridge: float = 0.01,
                 eps: float = 1e-8):
        self.basis = basis
        self.n = n
        self.alpha = alpha
        self.ridge = ridge
        self.eps = eps

        self.classes_: np.ndarray | None = None
        self.reconstructors_: dict | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DSGEFeatureExtractor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.reconstructors_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            if X_c.shape[0] < 2:
                raise ValueError(f"Class {c} has only {X_c.shape[0]} samples; "
                                 "need >= 2 to estimate moments.")
            recon = KunchenkoReconstructor(
                basis=self.basis, n=self.n, alpha=self.alpha, ridge=self.ridge
            ).fit_class(X_c)
            self.reconstructors_[c] = recon
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reconstructors_ is None:
            raise RuntimeError("Extractor not fitted; call fit first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        feats = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for j, c in enumerate(self.classes_):
            mse = self.reconstructors_[c].mse_per_sample(X)
            feats[:, j] = np.log(mse + self.eps)
        return feats

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def conditioning_report(self) -> dict:
        """Per-class cond(F_reg) — useful diagnostic.

        Values > 1e6 suggest increasing `ridge`, reducing `n`, or trying a
        different basis.
        """
        if self.reconstructors_ is None:
            return {}
        return {c: r.cond_F for c, r in self.reconstructors_.items()}


def _pool_points(X: np.ndarray, d: int) -> np.ndarray:
    """Flatten observations to a pooled (M, d) point cloud.

    Accepts a single observation set (n_obs, d) or a batch of samples
    (n_samples, n_obs, d); both pool to (M, d) for moment estimation.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        if X.shape[1] != d:
            raise ValueError(f"Expected last axis = d = {d}, got shape {X.shape}.")
        return X
    if X.ndim == 3:
        if X.shape[2] != d:
            raise ValueError(f"Expected last axis = d = {d}, got shape {X.shape}.")
        return X.reshape(-1, d)
    raise ValueError(f"Expected (n_obs, d) or (n_samples, n_obs, d), got ndim {X.ndim}.")


class MultivariateKunchenkoReconstructor:
    """Joint DSGE reconstruction for d-dimensional observations.

    Each observation is a point `x = (x_1, ..., x_d)`. Unlike the
    component-wise `KunchenkoReconstructor` (which pools all scalars and so
    cannot see how coordinates co-vary), this reconstructs *each coordinate*
    from the tensor-product basis of *all* coordinates. The Gram matrix `F`
    therefore carries cross-coordinate moments — the joint structure.

    All `d` output coordinates share the same `F` and differ only in the
    right-hand side (one `B` per coordinate):

        F[a,b]   = E[(Phi_a - psi_a)(Phi_b - psi_b)]      (D x D)
        B_k[a]   = E[(x_k - psi0_k)(Phi_a - psi_a)]       (D,)  for coord k
        F K_k    = B_k                                    K is (D, d)

    Input shape: (n_obs, d) for a single set, or (n_samples, n_obs, d) for a
    batch (pooled when fitting). `d` is inferred from `fit_class` and fixed.
    """

    def __init__(self, basis: str = "fractional", n: int = 3, d: int = 2,
                 alpha: float | None = None, ridge: float = 0.01):
        self.basis_name = basis
        self.n = n
        self.d = d
        self.alpha = alpha
        self.ridge = ridge
        self.tpb = TensorProductBasis(basis=basis, n=n, d=d, alpha=alpha)
        self.D = self.tpb.dim

        self.K: np.ndarray | None = None        # (D, d)
        self.k0: np.ndarray | None = None        # (d,)
        self.psi_0: np.ndarray | None = None     # (d,)
        self.psi: np.ndarray | None = None       # (D,)
        self.cond_F: float | None = None

    def fit_class(self, X_class: np.ndarray) -> "MultivariateKunchenkoReconstructor":
        """Fit the joint reconstructor on pooled observations from one class."""
        pts = _pool_points(X_class, self.d)        # (M, d)
        Phi = self.tpb.evaluate(pts)               # (M, D)
        M = pts.shape[0]

        self.psi_0 = pts.mean(axis=0)              # (d,)
        self.psi = Phi.mean(axis=0)                # (D,)

        Phi_c = Phi - self.psi                     # (M, D)
        X_c = pts - self.psi_0                      # (M, d)

        F = (Phi_c.T @ Phi_c) / M                  # (D, D)
        B = (Phi_c.T @ X_c) / M                    # (D, d) — one column per coord
        F_reg = F + self.ridge * np.eye(self.D)
        self.cond_F = float(np.linalg.cond(F_reg))

        self.K = np.linalg.solve(F_reg, B)         # (D, d)
        self.k0 = self.psi_0 - self.K.T @ self.psi  # (d,)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted reconstruction; preserves input shape (..., d)."""
        if self.K is None:
            raise RuntimeError("Reconstructor not fitted; call fit_class first.")
        X = np.asarray(X, dtype=np.float64)
        original_shape = X.shape
        pts = X.reshape(-1, self.d)
        Phi = self.tpb.evaluate(pts)
        X_hat = self.k0 + Phi @ self.K             # (M, d)
        return X_hat.reshape(original_shape)

    def mse_per_sample(self, X: np.ndarray) -> np.ndarray:
        """Per-sample joint MSE, averaged over observations and coordinates.

        X of shape (n_samples, n_obs, d) -> (n_samples,). A single observation
        set (n_obs, d) is treated as one sample.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None]
        n_samples, n_obs, d = X.shape
        if d != self.d:
            raise ValueError(f"Expected last axis = d = {self.d}, got {d}.")
        X_hat = self.reconstruct(X.reshape(-1, d)).reshape(n_samples, n_obs, d)
        return np.mean((X - X_hat) ** 2, axis=(1, 2))


class MultivariateDSGEFeatureExtractor:
    """Per-class joint reconstruction with log-MSED features (d-variate).

    The multivariate analogue of `DSGEFeatureExtractor`: one
    `MultivariateKunchenkoReconstructor` per class, log-MSED feature matrix
    of shape (n_samples, n_classes). Use when each sample is a population of
    d-dimensional observations and the *joint* distribution (not just the
    marginals) is class-discriminative.

    Input X: (n_samples, n_obs, d). `d` is taken from `X.shape[-1]`.
    """

    def __init__(self, basis: str = "fractional", n: int = 3,
                 alpha: float | None = None, ridge: float = 0.01,
                 eps: float = 1e-8):
        self.basis = basis
        self.n = n
        self.alpha = alpha
        self.ridge = ridge
        self.eps = eps

        self.d_: int | None = None
        self.classes_: np.ndarray | None = None
        self.reconstructors_: dict | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultivariateDSGEFeatureExtractor":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"Expected X of shape (n_samples, n_obs, d), got ndim {X.ndim}.")
        y = np.asarray(y)
        self.d_ = X.shape[2]
        self.classes_ = np.unique(y)
        self.reconstructors_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            if X_c.shape[0] < 2:
                raise ValueError(f"Class {c} has only {X_c.shape[0]} samples; "
                                 "need >= 2 to estimate moments.")
            recon = MultivariateKunchenkoReconstructor(
                basis=self.basis, n=self.n, d=self.d_,
                alpha=self.alpha, ridge=self.ridge,
            ).fit_class(X_c)
            self.reconstructors_[c] = recon
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reconstructors_ is None:
            raise RuntimeError("Extractor not fitted; call fit first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"Expected X of shape (n_samples, n_obs, d), got ndim {X.ndim}.")
        feats = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for j, c in enumerate(self.classes_):
            mse = self.reconstructors_[c].mse_per_sample(X)
            feats[:, j] = np.log(mse + self.eps)
        return feats

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def conditioning_report(self) -> dict:
        """Per-class cond(F_reg). Values > 1e6 warrant a larger `ridge`,
        smaller `n`, or a smaller `d`."""
        if self.reconstructors_ is None:
            return {}
        return {c: r.cond_F for c, r in self.reconstructors_.items()}
