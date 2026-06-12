"""Direction-of-arrival estimation via differential amplitude (Phase 7 Stream A).

    differential   — per-burst amplitude on each channel + Δ_dB ± CI
    averaging      — rolling median over Δ_dB stream + Newey-West HAC SE,
                      optional per-fc-hop bucketing
    null_detector  — flag the "null" when |Δ̄| stays in band; also exports
                      `theta_from_delta_db(geometry)` to map Δ̄ → θ̂.
"""

from .averaging import Averager, AveragerSummary, offline_summary
from .differential import (
    AmplitudeEstimator,
    DifferentialEstimate,
    DifferentialEstimator,
    amplitude_mean,
    amplitude_pmm2_with_fallback,
)
from .null_detector import (
    NullDetector,
    predicted_delta_db_grid,
    theta_from_delta_db,
)

__all__ = [
    "AmplitudeEstimator",
    "Averager",
    "AveragerSummary",
    "DifferentialEstimate",
    "DifferentialEstimator",
    "NullDetector",
    "amplitude_mean",
    "amplitude_pmm2_with_fallback",
    "offline_summary",
    "predicted_delta_db_grid",
    "theta_from_delta_db",
]
