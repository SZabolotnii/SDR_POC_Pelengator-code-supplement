"""Mavic-vs-rest SEI fingerprint filter (Phase 7 / A3, Plan-B primary path).

After A1 ruled out a deterministic OcuSync-1 header byte (see
docs/RESEARCH_OCUSYNC1_HEADER.md), the pelengator pipeline relies on a
DSGE+spectral fingerprint to single out the target controller's packets.
This is the productionised wrapper around the Phase 6 hybrid stack
(`fractional` DSGE on Re/Im branches + log-PSD geometric bins +
LogisticRegression) that hit 99.83 % on DJI vs non-DJI in
experiments/phase6_tampere_dsge.

Train on per-window IQ data (default 4096 complex64 samples, z-scored
per window). Predict either per window (`predict_proba`) or aggregate
across the windows of a single burst (`predict_burst_proba`). Decision
threshold is tunable post-fit so we can trade precision for recall to
hit the plan §A3 acceptance (precision ≥ 0.95, recall ≥ 0.85).

External dependency
-------------------
The complex-valued frac DSGE extractor lives in the dsge-toolkit
repository (~/Project/Research/DSGE/dsge-toolkit/scripts/) — same code
Phase 6 used. We import it lazily through `_import_complex_dsge` so that
machines without the toolkit still get a clear error message instead of
a stack trace at import time. Porting the extractor into
`src/sdr_kunchenko/dsge/` is tracked as deferred follow-up — for the POC
we accept the soft dependency.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Code-supplement note: the only edit relative to the research repository.
# The frac-DSGE extractor is vendored under vendor/dsge-toolkit/scripts/
# so reviewers do not need the external dsge-toolkit checkout.
DEFAULT_DSGE_TOOLKIT = (
    Path(__file__).resolve().parents[3] / "vendor" / "dsge-toolkit" / "scripts"
)


# ---------------------------------------------------------------- helpers


def _import_complex_dsge(toolkit_path: Path | None = None) -> type:
    """Lazy-import `ComplexDSGEFeatureExtractor` from external dsge-toolkit.

    Adds `toolkit_path` (or the auto-detected default) to `sys.path` once
    and returns the class. Raises `ImportError` with explicit instructions
    if the toolkit cannot be located.
    """
    p = Path(toolkit_path) if toolkit_path else DEFAULT_DSGE_TOOLKIT
    if not p.exists():
        raise ImportError(
            f"dsge-toolkit not found at {p}. Pass `dsge_toolkit_path=` to "
            f"FingerprintFilter or place the toolkit at "
            f"{DEFAULT_DSGE_TOOLKIT}. (Used by Phase 6 hybrid pipeline.)"
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    from complex_features import ComplexDSGEFeatureExtractor  # noqa: PLC0415
    return ComplexDSGEFeatureExtractor


def spectral_log_psd_features(X: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Geometric-bin log-PSD per window — same recipe as Phase 6.

    Parameters
    ----------
    X : (N, T) complex array
        Each row is one IQ window.
    n_bins : int
        Target number of geometric bins (the actual count may be smaller
        if duplicate edge integers collapse).

    Returns
    -------
    (N, n_bins_actual) float array of log10(mean PSD per bin).
    """
    if X.ndim != 2:
        raise ValueError(f"expected (N, T) windows, got shape {X.shape}")
    psd = np.abs(np.fft.fft(X, axis=1)) ** 2 / X.shape[1]
    psd = psd[:, : X.shape[1] // 2]
    edges = np.unique(np.geomspace(1, psd.shape[1], n_bins + 1).astype(int))
    bins = np.zeros((X.shape[0], len(edges) - 1), dtype=np.float64)
    for i in range(len(edges) - 1):
        bins[:, i] = psd[:, edges[i] : edges[i + 1]].mean(axis=1)
    return np.log10(bins + 1e-12)


def windowize_burst(
    iq_burst: np.ndarray, window_size: int, *, stride: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Cut one burst's IQ stream into fixed-size complex windows.

    Returns (M, window_size) complex64. M can be 0 if the burst is shorter
    than `window_size`. Per-window z-score normalisation matches the
    Phase 6 training-time preprocessing.
    """
    iq = np.asarray(iq_burst).ravel()
    if iq.size < window_size:
        return np.empty((0, window_size), dtype=np.complex64)
    step = stride or window_size
    n_win = 1 + (iq.size - window_size) // step
    starts = np.arange(n_win) * step
    out = np.stack([iq[s : s + window_size] for s in starts])
    if normalize:
        mean = out.mean(axis=1, keepdims=True)
        std = out.std(axis=1, keepdims=True) + 1e-9
        out = (out - mean) / std
    return out.astype(np.complex64)


# ---------------------------------------------------------------- config


@dataclass
class FingerprintConfig:
    """Tunable knobs for the FingerprintFilter pipeline."""
    window_size: int = 4096            # Phase 6 default
    n_spectral_bins: int = 16
    dsge_basis: str = "fractional"
    dsge_mode: str = "re_im"
    dsge_n: int = 3
    dsge_ridge: float = 0.01
    lr_max_iter: int = 2000
    lr_C: float = 1.0
    threshold: float = 0.5


# ---------------------------------------------------------------- core


class FingerprintFilter:
    """Binary 'is target Mavic Pro' classifier on per-window IQ.

    Convention: y = 1 → target class (Mavic Pro), y = 0 → any other drone.
    """

    def __init__(
        self, *, config: FingerprintConfig | None = None,
        dsge_toolkit_path: Path | None = None,
    ):
        self.cfg: FingerprintConfig = config or FingerprintConfig()
        self._toolkit_path = dsge_toolkit_path
        self._dsge: Any | None = None
        self._scaler: StandardScaler | None = None
        self._clf: LogisticRegression | None = None
        self._fit_metadata: dict = {}

    # ----- training

    def fit(self, X: np.ndarray, y: np.ndarray) -> FingerprintFilter:
        """Fit the DSGE extractor + scaler + LR on per-window data.

        Parameters
        ----------
        X : (N, T) complex array
            Each row is one z-scored IQ window of length `window_size`.
        y : (N,) int array
            Binary labels in {0, 1}; 1 = Mavic Pro.
        """
        if X.ndim != 2 or not np.iscomplexobj(X):
            raise ValueError(f"X must be (N, T) complex, got {X.shape} {X.dtype}")
        if X.shape[1] != self.cfg.window_size:
            raise ValueError(
                f"window_size mismatch: X has {X.shape[1]}, cfg has {self.cfg.window_size}"
            )
        y = np.asarray(y).astype(int)
        labels = np.unique(y)
        if not np.array_equal(labels, [0, 1]):
            raise ValueError(f"y must contain both 0 and 1, got labels {labels}")

        Cls = _import_complex_dsge(self._toolkit_path)
        self._dsge = Cls(
            mode=self.cfg.dsge_mode, basis=self.cfg.dsge_basis,
            n=self.cfg.dsge_n, ridge=self.cfg.dsge_ridge,
        ).fit(X, y)

        feat = self._build_features(X)
        self._scaler = StandardScaler().fit(feat)
        self._clf = LogisticRegression(
            max_iter=self.cfg.lr_max_iter, C=self.cfg.lr_C,
        ).fit(self._scaler.transform(feat), y)

        self._fit_metadata = {
            "n_train_windows": int(X.shape[0]),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            "feature_dim": int(feat.shape[1]),
            "dsge_dim": int(feat.shape[1] - self._psd_dim(X.shape[1])),
            "psd_dim": int(self._psd_dim(X.shape[1])),
        }
        return self

    def _psd_dim(self, T: int) -> int:
        edges = np.unique(
            np.geomspace(1, T // 2, self.cfg.n_spectral_bins + 1).astype(int)
        )
        return len(edges) - 1

    def _build_features(self, X: np.ndarray) -> np.ndarray:
        if self._dsge is None:
            raise RuntimeError("DSGE extractor not initialised")
        dsge_feat = self._dsge.transform(X)
        psd_feat = spectral_log_psd_features(X, n_bins=self.cfg.n_spectral_bins)
        return np.hstack([dsge_feat, psd_feat])

    # ----- prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-window P(target=Mavic Pro)."""
        self._require_fitted()
        feat = self._build_features(X)
        return self._clf.predict_proba(self._scaler.transform(feat))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.cfg.threshold).astype(int)

    def predict_burst_proba(
        self, iq_burst: np.ndarray, *, stride: int | None = None,
        aggregate: str = "mean",
    ) -> float:
        """Aggregate per-window probabilities for a single burst.

        `aggregate` ∈ {"mean", "max", "median"}. If the burst is shorter
        than `window_size` the function returns 0.0 (cannot fingerprint).
        """
        wins = windowize_burst(iq_burst, self.cfg.window_size, stride=stride)
        if wins.size == 0:
            return 0.0
        probs = self.predict_proba(wins)
        if aggregate == "mean":
            return float(probs.mean())
        if aggregate == "max":
            return float(probs.max())
        if aggregate == "median":
            return float(np.median(probs))
        raise ValueError(f"unknown aggregate {aggregate!r}")

    # ----- threshold + persistence

    def set_threshold(self, t: float) -> None:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {t}")
        self.cfg.threshold = float(t)

    def fit_metadata(self) -> dict:
        return dict(self._fit_metadata)

    def save(self, path: str | Path) -> None:
        """Persist via cloudpickle. The DSGE extractor must be importable
        again on `load` — pass `dsge_toolkit_path` if it is non-default.

        We use cloudpickle (not joblib) because the external DSGE basis
        functions are constructed as closures (lambdas), which the stock
        pickle protocol cannot serialise.
        """
        self._require_fitted()
        with open(path, "wb") as f:
            cloudpickle.dump(
                {
                    "cfg": self.cfg, "dsge": self._dsge,
                    "scaler": self._scaler, "clf": self._clf,
                    "metadata": self._fit_metadata,
                },
                f,
            )

    @classmethod
    def load(
        cls, path: str | Path, *, dsge_toolkit_path: Path | None = None,
    ) -> FingerprintFilter:
        # importing the class registers it for unpickling
        _import_complex_dsge(dsge_toolkit_path)
        with open(path, "rb") as f:
            d = cloudpickle.load(f)
        ff = cls(config=d["cfg"], dsge_toolkit_path=dsge_toolkit_path)
        ff._dsge = d["dsge"]
        ff._scaler = d["scaler"]
        ff._clf = d["clf"]
        ff._fit_metadata = d.get("metadata", {})
        return ff

    # ----- internal

    def _require_fitted(self) -> None:
        if self._clf is None or self._scaler is None or self._dsge is None:
            raise RuntimeError("FingerprintFilter must be fit() first")


__all__ = [
    "DEFAULT_DSGE_TOOLKIT",
    "FingerprintConfig",
    "FingerprintFilter",
    "spectral_log_psd_features",
    "windowize_burst",
]
