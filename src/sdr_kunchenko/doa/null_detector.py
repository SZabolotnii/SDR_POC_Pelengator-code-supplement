"""Null detector and Δ_dB → θ̂ inverse map (Phase 7 / A4).

A `NullDetector` wraps the `AveragerSummary` stream from `averaging.py` and
flags the moment the rolling estimate sits inside a tight band around 0 dB:

    |median_db| < abs_db_thr  AND  ci_width_db < ci_width_thr
    for `min_consecutive` updates in a row

Defaults match the plan: 0.5 dB centre, 1.0 dB CI width, 3 updates ≈ 3 s
@ 1 packet/s.

`theta_from_delta_db` inverts the geometry: given a measured Δ̄_dB and an
`AntennaGeometry`, do a 1-D grid search over the cos² beam pattern to
return the θ̂ whose predicted Δ matches the measurement most closely.
This is what closes the loop in A5 (and the A4 acceptance check).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rf.dual_channel_simulator import AntennaGeometry, cosine_beam_pattern
from .averaging import AveragerSummary

# ---------------------------------------------------------------- detector


@dataclass
class NullDetector:
    """Streaming flag: target lies "in the null" of the differential beam."""
    abs_db_thr: float = 0.5
    ci_width_thr: float = 1.0
    min_consecutive: int = 3

    _consec: int = 0

    def update(self, summary: AveragerSummary) -> bool:
        """Returns True iff the trigger condition has been met for ≥ k updates.

        NaNs (typically when the averager is still empty) reset the counter.
        """
        med = summary.median_db
        ci_w = summary.ci_width_db
        if not (np.isfinite(med) and np.isfinite(ci_w)):
            self._consec = 0
            return False
        in_band = abs(med) < self.abs_db_thr and ci_w < self.ci_width_thr
        self._consec = self._consec + 1 if in_band else 0
        return self._consec >= self.min_consecutive

    @property
    def consecutive(self) -> int:
        return self._consec

    def reset(self) -> None:
        self._consec = 0


# ------------------------------------------------------- inverse mapping


def predicted_delta_db_grid(
    geometry: AntennaGeometry, *, theta_grid_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate Δ_dB(θ) on a grid for the given antenna geometry.

    Returns (theta_grid_deg, delta_db_grid). Where both gains are at the
    floor the table is exactly 0 dB and provides no DOA information; that
    region (the "blind sector") is the user's responsibility to gate.
    """
    if theta_grid_deg is None:
        theta_grid_deg = np.arange(-89.0, 89.0001, 0.1)
    g_a = cosine_beam_pattern(
        theta_grid_deg, geometry.boresight_a_deg, geometry.beamwidth_3db_deg,
        floor_db=geometry.pattern_floor_db,
    )
    g_b = cosine_beam_pattern(
        theta_grid_deg, geometry.boresight_b_deg, geometry.beamwidth_3db_deg,
        floor_db=geometry.pattern_floor_db,
    )
    delta_db = 10.0 * np.log10(g_a / g_b)
    return theta_grid_deg.astype(np.float64), delta_db.astype(np.float64)


def theta_from_delta_db(
    delta_db: float, geometry: AntennaGeometry, *,
    theta_grid_deg: np.ndarray | None = None,
    restrict_active_sector: bool = True,
) -> float:
    """Return θ̂ in degrees s.t. Δ_dB(θ̂) ≈ measured Δ.

    With `restrict_active_sector=True` the search is clipped to angles
    where at least one antenna is *not* on the floor — beyond that the
    inverse becomes degenerate (any blind θ predicts Δ ≈ 0 dB). The
    caller can override by passing a custom grid.
    """
    grid, dgrid = predicted_delta_db_grid(geometry, theta_grid_deg=theta_grid_deg)
    if restrict_active_sector:
        floor_db = geometry.pattern_floor_db + 0.1   # tiny margin
        g_a = cosine_beam_pattern(
            grid, geometry.boresight_a_deg, geometry.beamwidth_3db_deg,
            floor_db=geometry.pattern_floor_db,
        )
        g_b = cosine_beam_pattern(
            grid, geometry.boresight_b_deg, geometry.beamwidth_3db_deg,
            floor_db=geometry.pattern_floor_db,
        )
        floored = (10 * np.log10(g_a) <= floor_db) & (10 * np.log10(g_b) <= floor_db)
        active = ~floored
        if active.any():
            grid = grid[active]
            dgrid = dgrid[active]
    idx = int(np.argmin(np.abs(dgrid - float(delta_db))))
    return float(grid[idx])


__all__ = [
    "NullDetector",
    "predicted_delta_db_grid",
    "theta_from_delta_db",
]
