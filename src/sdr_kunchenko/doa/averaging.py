"""Rolling-window averaging of per-burst Δ_dB with optional per-hop bucketing.

Phase 7 / A4 — companion to `DifferentialEstimator`. Holds the last `window`
estimates in a deque, exposes a `summary()` dict that the `NullDetector`
reads. Variance is reported as Newey-West HAC SE on the *median* statistic,
which is robust to fc-hop outliers and matches the spec requirement of
"ковзна median + Newey-West HAC SE через rolling window 200 пакетів".

If `bucket_by_hop=True`, the averager groups recent estimates by their
`fc_offset_hz` (rounded to `bucket_step_hz`) so a hopping OcuSync stream
produces per-hop sub-medians; the top-level `summary()` then weights each
bucket equally, reducing single-frequency-channel multipath bias.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .differential import DifferentialEstimate

_Z95 = 1.959964


@dataclass(frozen=True)
class AveragerSummary:
    """Aggregate over the rolling window."""
    n: int
    median_db: float
    se_db: float
    ci95_db: tuple[float, float]
    n_buckets: int
    estimator_names: tuple[str, ...]

    @property
    def ci_width_db(self) -> float:
        return self.ci95_db[1] - self.ci95_db[0]

    def to_dict(self) -> dict:
        return {
            "n": int(self.n),
            "n_buckets": int(self.n_buckets),
            "median_db": float(self.median_db),
            "se_db": float(self.se_db),
            "ci95_low_db": float(self.ci95_db[0]),
            "ci95_high_db": float(self.ci95_db[1]),
            "estimator_names": list(self.estimator_names),
        }


@dataclass
class _Item:
    delta_db: float
    se_db: float
    fc_offset_hz: float | None
    estimator: str


def _newey_west_se(x: np.ndarray, lag: int) -> float:
    """HAC SE estimator for the sample mean of x.

    s² = γ₀ + 2 · Σ_{ℓ=1..lag} (1 − ℓ/(lag+1)) · γ_ℓ
    SE = √(s²/n)
    Bartlett window weights with truncation `lag`. If `lag ≥ n` we clamp.
    """
    n = x.size
    if n < 2:
        return float("nan")
    lag = max(0, min(lag, n - 1))
    z = x - x.mean()
    gamma_0 = float(np.dot(z, z) / n)
    s2 = gamma_0
    for ell in range(1, lag + 1):
        w = 1.0 - ell / (lag + 1.0)
        gamma_l = float(np.dot(z[:-ell], z[ell:]) / n)
        s2 += 2.0 * w * gamma_l
    return float(np.sqrt(max(s2, 0.0) / n))


@dataclass
class Averager:
    """Sliding-window aggregator over `DifferentialEstimate` instances."""
    window: int = 200
    hac_lag: int = 5
    bucket_by_hop: bool = False
    bucket_step_hz: float = 1.0e6

    _buf: deque = field(default_factory=deque, init=False, repr=False)
    _names: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")
        # rebind deque with the configured maxlen (default_factory builds an empty one)
        self._buf = deque(maxlen=self.window)

    def push(
        self, estimate: DifferentialEstimate, *, fc_offset_hz: float | None = None,
    ) -> AveragerSummary:
        item = _Item(
            delta_db=float(estimate.delta_db),
            se_db=float(estimate.se_db),
            fc_offset_hz=None if fc_offset_hz is None else float(fc_offset_hz),
            estimator=estimate.estimator_name,
        )
        self._buf.append(item)
        self._names.add(item.estimator)
        return self.summary()

    def reset(self) -> None:
        self._buf.clear()
        self._names.clear()

    @property
    def n(self) -> int:
        return len(self._buf)

    def summary(self) -> AveragerSummary:
        if not self._buf:
            return AveragerSummary(
                n=0, median_db=float("nan"), se_db=float("nan"),
                ci95_db=(float("nan"), float("nan")),
                n_buckets=0, estimator_names=tuple(self._names),
            )

        deltas = np.array([it.delta_db for it in self._buf], dtype=np.float64)

        if self.bucket_by_hop and any(it.fc_offset_hz is not None for it in self._buf):
            buckets = self._collect_buckets()
            per_bucket_medians = np.array(
                [np.median(b) for b in buckets.values()], dtype=np.float64
            )
            median_db = float(np.median(per_bucket_medians))
            se = _newey_west_se(per_bucket_medians, self.hac_lag)
            n_buckets = len(buckets)
        else:
            median_db = float(np.median(deltas))
            se = _newey_west_se(deltas, self.hac_lag)
            n_buckets = 1

        ci_low = median_db - _Z95 * se
        ci_high = median_db + _Z95 * se
        return AveragerSummary(
            n=len(self._buf),
            median_db=median_db,
            se_db=float(se),
            ci95_db=(float(ci_low), float(ci_high)),
            n_buckets=n_buckets,
            estimator_names=tuple(sorted(self._names)),
        )

    def _collect_buckets(self) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for it in self._buf:
            if it.fc_offset_hz is None:
                key = 0
            else:
                key = int(round(it.fc_offset_hz / self.bucket_step_hz))
            out.setdefault(key, []).append(it.delta_db)
        return out


def offline_summary(
    estimates: Sequence[DifferentialEstimate], *,
    fc_offsets_hz: Sequence[float | None] | None = None,
    window: int = 200, hac_lag: int = 5,
    bucket_by_hop: bool = False, bucket_step_hz: float = 1.0e6,
) -> AveragerSummary:
    """Convenience: feed a sequence of estimates through an Averager once."""
    avg = Averager(
        window=window, hac_lag=hac_lag,
        bucket_by_hop=bucket_by_hop, bucket_step_hz=bucket_step_hz,
    )
    if fc_offsets_hz is None:
        fc_offsets_hz = [None] * len(estimates)
    if len(fc_offsets_hz) != len(estimates):
        raise ValueError("fc_offsets_hz length mismatch")
    for est, fc in zip(estimates, fc_offsets_hz, strict=True):
        avg.push(est, fc_offset_hz=fc)
    return avg.summary()


__all__ = [
    "Averager",
    "AveragerSummary",
    "offline_summary",
]
