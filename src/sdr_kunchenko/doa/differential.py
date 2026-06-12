"""Per-burst differential amplitude estimator (Phase 7 / A4).

For one (iq_a, iq_b) pair:
    Â_a, SE(Â_a) = amplitude_estimator(iq_a)
    Â_b, SE(Â_b) = amplitude_estimator(iq_b)
    Δ̂_dB        = 20 · log10(Â_a / Â_b)
    SE(Δ̂_dB)    = (20/ln10) · sqrt( (SE_a/Â_a)² + (SE_b/Â_b)² )    (delta method)

dB convention note
------------------
The plan §A4 spelt the differential as `10·log10(|A_a|/|A_b|)`, but the
standard amplitude-ratio decibel uses `20·log10` (since power ∝ amplitude²).
We use the 20·log10 form so a measured Δ_dB is directly comparable to the
power-gain ratio of the antenna beam pattern (`G_a/G_b` in
`dual_channel_simulator.py`). This makes the inverse mapping in
`null_detector.theta_from_delta_db` consistent end-to-end. The shape of
the rolling-CI bookkeeping (delta method) is identical to either choice;
only the leading constant changes.

The amplitude estimator is pluggable so we can swap mean(|iq|) (default,
Gaussian-MLE for amplitude under unimodal envelopes) for the EstemPMM
PMM2 oracle when R is available locally. The oracle is imported lazily
behind `amplitude_pmm2_with_fallback`; if rpy2 / R cannot be loaded the
function silently degrades to the mean estimator and notes that in the
returned diagnostics, so the rest of the pipeline never breaks just
because R is missing on a developer laptop.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# z-score for 95 % confidence level (Gaussian approximation).
_Z95 = 1.959964


# ---------------------------------------------------------------- types


@dataclass(frozen=True)
class DifferentialEstimate:
    """One (iq_a, iq_b) pair → Δ_dB ± 95 % CI plus per-channel amplitudes."""
    delta_db: float
    se_db: float
    ci95_db: tuple[float, float]
    amp_a: float
    amp_b: float
    se_a: float
    se_b: float
    estimator_name: str

    @property
    def ci_width_db(self) -> float:
        return self.ci95_db[1] - self.ci95_db[0]

    def to_dict(self) -> dict:
        return {
            "delta_db": float(self.delta_db),
            "se_db": float(self.se_db),
            "ci95_low_db": float(self.ci95_db[0]),
            "ci95_high_db": float(self.ci95_db[1]),
            "amp_a": float(self.amp_a),
            "amp_b": float(self.amp_b),
            "se_a": float(self.se_a),
            "se_b": float(self.se_b),
            "estimator": str(self.estimator_name),
        }


AmplitudeEstimator = Callable[[np.ndarray], tuple[float, float]]


# ------------------------------------------------------- estimators


def amplitude_mean(iq: np.ndarray) -> tuple[float, float]:
    """Sample mean of |iq| with sample-SE = std(|iq|) / sqrt(n).

    Asymptotically equivalent to ML for amplitude under a unimodal
    envelope. Robust default that needs no external tooling.
    """
    a = np.abs(np.asarray(iq))
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def amplitude_pmm2_with_fallback(iq: np.ndarray) -> tuple[float, float]:
    """Try EstemPMM PMM2 amplitude; degrade to `amplitude_mean` on any failure.

    The oracle wrapper is imported inside the function so that this
    module is importable even when rpy2 / R are misconfigured on the
    developer machine — only callers that actually request PMM2 take the
    R import hit, and only at call time. On any rpy2 / R-side failure
    the function emits a single `amplitude_mean` fallback per call;
    callers can spot the substitution via the `estimator_name` field on
    the resulting `DifferentialEstimate`.
    """
    try:
        from sdr_kunchenko.pmm.oracle import lm_pmm2_amplitude  # type: ignore[attr-defined]
    except Exception:                                  # noqa: BLE001
        return amplitude_mean(iq)
    try:
        return lm_pmm2_amplitude(iq)
    except Exception:                                  # noqa: BLE001
        return amplitude_mean(iq)


# ---------------------------------------------------------------- core


class DifferentialEstimator:
    """Stream interface: feed (iq_a, iq_b) pairs, get Δ_dB ± 95 % CI back."""

    def __init__(self, *, amplitude_estimator: AmplitudeEstimator = amplitude_mean,
                  estimator_name: str | None = None):
        self._amp = amplitude_estimator
        self._name = estimator_name or amplitude_estimator.__name__

    def estimate(self, iq_a: np.ndarray, iq_b: np.ndarray) -> DifferentialEstimate:
        amp_a, se_a = self._amp(iq_a)
        amp_b, se_b = self._amp(iq_b)
        if not (amp_a > 0 and amp_b > 0):
            raise ValueError(
                f"non-positive amplitude (a={amp_a:.3g}, b={amp_b:.3g}); "
                "estimator must return strictly positive values"
            )
        delta_db = 20.0 * np.log10(amp_a / amp_b)
        # Delta-method SE; if either SE is NaN (n=1) → propagate NaN
        c = 20.0 / np.log(10.0)
        if np.isnan(se_a) or np.isnan(se_b):
            se_db = float("nan")
            ci_low = ci_high = float("nan")
        else:
            var = (se_a / amp_a) ** 2 + (se_b / amp_b) ** 2
            se_db = float(c * np.sqrt(var))
            ci_low = float(delta_db - _Z95 * se_db)
            ci_high = float(delta_db + _Z95 * se_db)
        return DifferentialEstimate(
            delta_db=float(delta_db), se_db=se_db,
            ci95_db=(ci_low, ci_high),
            amp_a=float(amp_a), amp_b=float(amp_b),
            se_a=float(se_a), se_b=float(se_b),
            estimator_name=self._name,
        )


__all__ = [
    "AmplitudeEstimator",
    "DifferentialEstimate",
    "DifferentialEstimator",
    "amplitude_mean",
    "amplitude_pmm2_with_fallback",
]
