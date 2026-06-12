"""Unit tests for the Mavic-vs-rest FingerprintFilter (Phase 7 / A3).

Hard runtime requirement: external dsge-toolkit must be importable. We
detect it the same way `FingerprintFilter` does and skip the whole
module if it is missing — there is no point mocking it for these tests
because the whole point of the wrapper is the real DSGE pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sdr_kunchenko.rf.fingerprint import (
    DEFAULT_DSGE_TOOLKIT,
    FingerprintConfig,
    FingerprintFilter,
    spectral_log_psd_features,
    windowize_burst,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_DSGE_TOOLKIT.exists(),
    reason=f"dsge-toolkit not at {DEFAULT_DSGE_TOOLKIT}; A3 cannot run",
)


# --------------------------------------------------------------- helpers


def _synth_positive(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Heavy-tail Cauchy-like proxy for the Mavic Pro class."""
    x = rng.standard_cauchy((n, T)).clip(-30, 30)
    y = rng.standard_cauchy((n, T)).clip(-30, 30)
    return (x + 1j * y).astype(np.complex64)


def _synth_negative(n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian proxy for the non-Mavic class."""
    x = rng.standard_normal((n, T))
    y = rng.standard_normal((n, T))
    return (x + 1j * y).astype(np.complex64)


def _build_dataset(n_per_class: int, T: int, seed: int = 0):
    """Build (X, y) with the same per-window z-score that windowize_burst
    applies — so training-time and inference-time distributions match.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([_synth_positive(n_per_class, T, rng),
                   _synth_negative(n_per_class, T, rng)])
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-9
    X = ((X - mean) / std).astype(np.complex64)
    y = np.array([1] * n_per_class + [0] * n_per_class, dtype=int)
    return X, y


# ------------------------------------------------------- spectral helper


def test_spectral_log_psd_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((6, 1024))
         + 1j * rng.standard_normal((6, 1024))).astype(np.complex64)
    F = spectral_log_psd_features(X, n_bins=8)
    assert F.shape[0] == 6
    assert F.shape[1] >= 4  # bins may collapse but never below ~half
    assert np.isfinite(F).all()
    assert F.dtype == np.float64


def test_spectral_log_psd_rejects_1d():
    with pytest.raises(ValueError):
        spectral_log_psd_features(np.zeros(128, dtype=np.complex64))


# ------------------------------------------------------- windowize helper


def test_windowize_burst_basic():
    iq = (np.arange(8000) + 1j * np.arange(8000)).astype(np.complex64)
    wins = windowize_burst(iq, window_size=4096, normalize=False)
    assert wins.shape == (1, 4096)
    wins_strided = windowize_burst(iq, window_size=4096, stride=1000, normalize=False)
    assert wins_strided.shape[0] == 1 + (8000 - 4096) // 1000


def test_windowize_burst_short_input_returns_empty():
    iq = np.zeros(100, dtype=np.complex64)
    wins = windowize_burst(iq, window_size=4096)
    assert wins.shape == (0, 4096)


def test_windowize_burst_zscore_normalises():
    rng = np.random.default_rng(0)
    iq = rng.standard_normal(8192).astype(np.complex64)
    iq += 5.0  # bias
    wins = windowize_burst(iq, window_size=4096, normalize=True)
    assert wins.shape == (2, 4096)  # 8192 = 2 contiguous windows
    for w in wins:
        assert abs(w.mean()) < 1e-3
        assert abs(np.std(w) - 1.0) < 1e-2


# ------------------------------------------------------- end-to-end fit/predict


@pytest.fixture(scope="module")
def trained_filter():
    X, y = _build_dataset(n_per_class=30, T=2048, seed=0)
    cfg = FingerprintConfig(window_size=2048, n_spectral_bins=8)
    return FingerprintFilter(config=cfg).fit(X, y), X, y


def test_fit_metadata_is_populated(trained_filter):
    ff, X, y = trained_filter
    md = ff.fit_metadata()
    assert md["n_train_windows"] == 60
    assert md["n_positive"] == 30
    assert md["n_negative"] == 30
    assert md["feature_dim"] == md["dsge_dim"] + md["psd_dim"]


def test_predict_proba_separates_classes_on_easy_data(trained_filter):
    ff, X, y = trained_filter
    p = ff.predict_proba(X)
    pos = p[y == 1]
    neg = p[y == 0]
    # Cauchy vs Gaussian is trivially separable; every positive should
    # score above every negative.
    assert pos.min() > neg.max(), (
        f"overlap: pos.min={pos.min():.3f} neg.max={neg.max():.3f}"
    )


def test_predict_burst_proba_aggregates(trained_filter):
    ff, X, y = trained_filter
    # Concat several training-time positives — guarantees the burst
    # statistics match the trained-on distribution exactly.
    pos_windows = X[y == 1][:3]                   # (3, T)
    long_pos = pos_windows.reshape(-1).astype(np.complex64)
    p = ff.predict_burst_proba(long_pos)
    assert 0.0 <= p <= 1.0
    assert p > 0.5, f"positive burst aggregated proba={p:.3f}"
    p_max = ff.predict_burst_proba(long_pos, aggregate="max")
    assert p_max >= p


def test_predict_burst_proba_too_short_returns_zero(trained_filter):
    ff, _, _ = trained_filter
    iq = np.zeros(100, dtype=np.complex64)
    assert ff.predict_burst_proba(iq) == 0.0


def test_threshold_changes_predictions(trained_filter):
    ff, X, y = trained_filter
    ff.set_threshold(0.5)
    n_pos_default = int(ff.predict(X).sum())
    ff.set_threshold(0.99)
    n_pos_strict = int(ff.predict(X).sum())
    assert n_pos_strict <= n_pos_default
    ff.set_threshold(0.5)  # restore for other tests


def test_threshold_validation():
    ff = FingerprintFilter()
    with pytest.raises(ValueError):
        ff.set_threshold(1.5)


# ------------------------------------------------------- input validation


def test_fit_rejects_real_input():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 2048))  # real
    y = np.array([0, 1] * 5)
    cfg = FingerprintConfig(window_size=2048)
    with pytest.raises(ValueError):
        FingerprintFilter(config=cfg).fit(X, y)


def test_fit_rejects_window_size_mismatch():
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((10, 1024))
         + 1j * rng.standard_normal((10, 1024))).astype(np.complex64)
    y = np.array([0, 1] * 5)
    cfg = FingerprintConfig(window_size=2048)
    with pytest.raises(ValueError):
        FingerprintFilter(config=cfg).fit(X, y)


def test_fit_rejects_single_class():
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((10, 2048))
         + 1j * rng.standard_normal((10, 2048))).astype(np.complex64)
    y = np.zeros(10, dtype=int)
    cfg = FingerprintConfig(window_size=2048)
    with pytest.raises(ValueError):
        FingerprintFilter(config=cfg).fit(X, y)


# ------------------------------------------------------- save / load round-trip


def test_save_load_round_trip(tmp_path: Path, trained_filter):
    ff, X, _ = trained_filter
    p_before = ff.predict_proba(X)
    out = tmp_path / "ff.joblib"
    ff.save(out)
    ff2 = FingerprintFilter.load(out)
    p_after = ff2.predict_proba(X)
    assert np.allclose(p_before, p_after)
    assert ff2.fit_metadata() == ff.fit_metadata()
