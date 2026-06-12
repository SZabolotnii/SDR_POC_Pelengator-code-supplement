"""Complex-valued DSGE feature generation for I/Q baseband signals.

SDR signals are complex-valued (I + jQ). Naively flattening to
[Re_0, Im_0, Re_1, Im_1, ...] doubles feature dimensionality without
adding information; treating |y| alone discards phase. This module
provides three principled ways to handle complex inputs:

    mode='magnitude_phase'  — fit DSGE on |y| and arg(y) separately,
                              concatenate features. Captures both
                              amplitude and phase structure.

    mode='re_im'            — fit DSGE on Re(y) and Im(y) separately,
                              concatenate features. Preserves I/Q geometry,
                              good for transmitter fingerprinting where
                              IQ imbalance is a discriminative signal.

    mode='magnitude'        — fit DSGE on |y| only. Drops phase info but
                              halves feature size. Good when only amplitude
                              statistics matter (energy detection).

Public API:
    ComplexKunchenkoReconstructor(mode='re_im', basis='fractional', ...)
        .fit_class(X_class)
        .mse_per_sample(X)

    ComplexDSGEFeatureExtractor(mode='re_im', ...)
        .fit(X, y)
        .transform(X) -> (n_samples, n_classes * branches) feature matrix
                          where branches = 2 (re_im, magnitude_phase) or 1 (magnitude)

The "branches" expansion reflects that for re_im mode we get a separate
log-MSED per class for the real and imaginary part, doubling the feature
dimensionality but preserving full I/Q information.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kunchenko_features import (  # noqa: E402
    DSGEFeatureExtractor,
    KunchenkoReconstructor,
)


VALID_MODES = ("re_im", "magnitude_phase", "magnitude")


def _split_complex(X: np.ndarray, mode: str) -> tuple[np.ndarray, ...]:
    """Convert complex input to one or two real-valued branches.

    Parameters
    ----------
    X : array
        Complex array (any shape; treated component-wise) OR real array
        already viewed as I/Q. For real input, mode='magnitude' is the
        only non-trivial choice (magnitude == |x|).
    mode : str
        One of 're_im', 'magnitude_phase', 'magnitude'.

    Returns
    -------
    tuple of arrays — same shape as X, two branches for 're_im' and
    'magnitude_phase' modes, one for 'magnitude'.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Mode must be one of {VALID_MODES}, got {mode!r}.")
    X = np.asarray(X)
    if not np.iscomplexobj(X):
        # Real input — treat as already-real signal
        if mode == "magnitude":
            return (np.abs(X),)
        if mode == "re_im":
            return (X, np.zeros_like(X))
        # magnitude_phase
        return (np.abs(X), np.zeros_like(X))

    if mode == "re_im":
        return (X.real.astype(np.float64), X.imag.astype(np.float64))
    if mode == "magnitude":
        return (np.abs(X).astype(np.float64),)
    if mode == "magnitude_phase":
        return (np.abs(X).astype(np.float64), np.angle(X).astype(np.float64))
    raise AssertionError("unreachable")


class ComplexKunchenkoReconstructor:
    """Per-class reconstructor for complex-valued signals.

    Fits one independent KunchenkoReconstructor per real-valued branch
    (Re/Im or |.|/arg(.) or just |.|). Reconstruction error is summed
    over branches, giving a scalar MSE per sample.
    """

    def __init__(self, mode: str = "re_im", basis: str = "fractional",
                 n: int = 3, alpha: float | None = None, ridge: float = 0.01):
        if mode not in VALID_MODES:
            raise ValueError(f"Mode must be one of {VALID_MODES}, got {mode!r}.")
        self.mode = mode
        self.basis = basis
        self.n = n
        self.alpha = alpha
        self.ridge = ridge
        self.branch_recons_: list[KunchenkoReconstructor] = []

    def fit_class(self, X_class: np.ndarray) -> "ComplexKunchenkoReconstructor":
        branches = _split_complex(X_class, self.mode)
        self.branch_recons_ = []
        for branch in branches:
            recon = KunchenkoReconstructor(
                basis=self.basis, n=self.n,
                alpha=self.alpha, ridge=self.ridge
            ).fit_class(branch)
            self.branch_recons_.append(recon)
        return self

    def mse_per_sample(self, X: np.ndarray) -> np.ndarray:
        if not self.branch_recons_:
            raise RuntimeError("Reconstructor not fitted; call fit_class first.")
        branches = _split_complex(X, self.mode)
        # Sum MSE across branches — gives total reconstruction error
        mses = [recon.mse_per_sample(b) for recon, b in
                zip(self.branch_recons_, branches)]
        return np.sum(np.stack(mses, axis=0), axis=0)

    @property
    def cond_F_per_branch(self) -> list[float]:
        return [r.cond_F for r in self.branch_recons_]


class ComplexDSGEFeatureExtractor:
    """DSGE feature extractor for complex-valued multi-class data.

    Mode 're_im' produces (n_samples, n_classes * 2) features —
    separate log-MSED for Re-branch and Im-branch per class. This
    preserves I/Q geometry information (IQ imbalance, phase noise
    asymmetries) that's discriminative for transmitter fingerprinting.

    Mode 'magnitude_phase' similarly produces 2× features (one per
    each of |y|, arg(y) branches). Better when phase statistics are
    independent of magnitude — e.g. carrier frequency offset effects.

    Mode 'magnitude' produces (n_samples, n_classes) features as in
    the real-valued DSGEFeatureExtractor. Use only when you've
    confirmed that phase carries no discriminative signal.
    """

    def __init__(self, mode: str = "re_im", basis: str = "fractional",
                 n: int = 3, alpha: float | None = None, ridge: float = 0.01,
                 eps: float = 1e-8, separate_features: bool = True):
        """
        Parameters
        ----------
        separate_features : bool, default True
            If True, the feature vector contains a separate log-MSED per
            (class, branch) — preserving branch-specific information.
            If False, MSE is summed across branches first, then logged —
            giving (n_samples, n_classes) like the real-valued extractor.
            Use False for compactness, True for maximum discriminativity.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Mode must be one of {VALID_MODES}, got {mode!r}.")
        self.mode = mode
        self.basis = basis
        self.n = n
        self.alpha = alpha
        self.ridge = ridge
        self.eps = eps
        self.separate_features = separate_features

        self.classes_: np.ndarray | None = None
        self.reconstructors_: dict | None = None
        self._n_branches: int = (
            1 if mode == "magnitude" else 2
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ComplexDSGEFeatureExtractor":
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.reconstructors_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            if X_c.shape[0] < 2:
                raise ValueError(f"Class {c} has only {X_c.shape[0]} samples; "
                                 "need >= 2 to estimate moments.")
            recon = ComplexKunchenkoReconstructor(
                mode=self.mode, basis=self.basis, n=self.n,
                alpha=self.alpha, ridge=self.ridge
            ).fit_class(X_c)
            self.reconstructors_[c] = recon
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reconstructors_ is None:
            raise RuntimeError("Extractor not fitted; call fit first.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_classes = len(self.classes_)
        if self.separate_features and self._n_branches > 1:
            # Per-branch features: (n_samples, n_classes * n_branches)
            feats = np.zeros((X.shape[0], n_classes * self._n_branches))
            for j, c in enumerate(self.classes_):
                recon = self.reconstructors_[c]
                branches = _split_complex(X, self.mode)
                for b_idx, (branch_data, branch_recon) in enumerate(
                        zip(branches, recon.branch_recons_)):
                    mse = branch_recon.mse_per_sample(branch_data)
                    feats[:, j * self._n_branches + b_idx] = np.log(mse + self.eps)
            return feats

        # Combined: (n_samples, n_classes)
        feats = np.zeros((X.shape[0], n_classes))
        for j, c in enumerate(self.classes_):
            mse = self.reconstructors_[c].mse_per_sample(X)
            feats[:, j] = np.log(mse + self.eps)
        return feats

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def conditioning_report(self) -> dict:
        """Per-(class, branch) cond(F_reg). If branches > 1, returns nested dict."""
        if self.reconstructors_ is None:
            return {}
        if self._n_branches == 1:
            return {c: r.cond_F_per_branch[0] for c, r in self.reconstructors_.items()}
        # Nested: {class: [cond_branch_0, cond_branch_1]}
        return {c: r.cond_F_per_branch for c, r in self.reconstructors_.items()}


__all__ = [
    "ComplexKunchenkoReconstructor",
    "ComplexDSGEFeatureExtractor",
    "VALID_MODES",
]
