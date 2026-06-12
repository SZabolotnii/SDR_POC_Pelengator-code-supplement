"""Tests for the doa package (Phase 7 / A4).

Three modules under test:
    differential   — DifferentialEstimator (delta-method CI on Δ_dB)
    averaging      — Averager (rolling window, Newey-West HAC SE)
    null_detector  — NullDetector + theta_from_delta_db inverse map

Acceptance from the plan §A4 — θ_target ∈ {-60, -30, 0, 30, 60} →
recovered θ̂ within ±5° — is exercised end-to-end in the dedicated
script `experiments/poc_pelengator/run_a4_acceptance.py`. The unit
tests below pin the math of each component in isolation.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sdr_kunchenko.doa import (
    Averager,
    DifferentialEstimator,
    NullDetector,
    amplitude_mean,
    offline_summary,
    predicted_delta_db_grid,
    theta_from_delta_db,
)
from sdr_kunchenko.doa.averaging import AveragerSummary
from sdr_kunchenko.rf.dual_channel_simulator import (
    AntennaGeometry,
    cosine_beam_pattern,
    simulate_dual_channel,
)

# ---------------------------------------------------------------- helpers


@pytest.fixture
def carrier_iq() -> np.ndarray:
    n = 4096
    fs = 10e6
    f0 = 250e3
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * f0 * t).astype(np.complex128)


@pytest.fixture
def wide_geom() -> AntennaGeometry:
    """Acceptance-test geometry: ±60° still inside the active sector."""
    return AntennaGeometry(
        boresight_a_deg=30.0, boresight_b_deg=-30.0, beamwidth_3db_deg=60.0,
    )


# ---------------------------------------------------------------- amplitude


def test_amplitude_mean_unit_carrier():
    iq = np.exp(1j * np.linspace(0, 4 * np.pi, 1024)).astype(np.complex128)
    a, se = amplitude_mean(iq)
    assert a == pytest.approx(1.0, abs=1e-12)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_amplitude_mean_returns_nan_se_on_singleton():
    a, se = amplitude_mean(np.array([3 + 4j], dtype=np.complex128))
    assert a == pytest.approx(5.0)
    assert np.isnan(se)


# ---------------------------------------------------------------- differential


def test_differential_zero_when_balanced(carrier_iq):
    de = DifferentialEstimator()
    est = de.estimate(carrier_iq, carrier_iq)
    assert est.delta_db == pytest.approx(0.0, abs=1e-12)
    # CW carrier → SE_a = SE_b = 0 → SE_db = 0 → CI degenerate
    assert est.se_db == pytest.approx(0.0, abs=1e-12)
    assert est.estimator_name == "amplitude_mean"


def test_differential_factor_two_amplitude(carrier_iq):
    de = DifferentialEstimator()
    est = de.estimate(2.0 * carrier_iq, carrier_iq)
    # 20·log10(2) ≈ 6.020599 dB
    assert est.delta_db == pytest.approx(6.020599, abs=1e-3)


def test_differential_rejects_zero_amplitude():
    de = DifferentialEstimator()
    iq_zero = np.zeros(64, dtype=np.complex128)
    iq_one = np.ones(64, dtype=np.complex128)
    with pytest.raises(ValueError):
        de.estimate(iq_zero, iq_one)


def test_differential_se_nonzero_with_noise():
    rng = np.random.default_rng(0)
    n = 1024
    iq_a = (1.0 + rng.standard_normal(n) * 0.1
             + 1j * rng.standard_normal(n) * 0.1).astype(np.complex128)
    iq_b = (0.5 + rng.standard_normal(n) * 0.1
             + 1j * rng.standard_normal(n) * 0.1).astype(np.complex128)
    est = DifferentialEstimator().estimate(iq_a, iq_b)
    assert est.se_db > 0
    assert est.ci_width_db > 0
    assert est.ci95_db[0] < est.delta_db < est.ci95_db[1]


# ---------------------------------------------------------------- averager


def test_averager_empty_summary_is_nan():
    s = Averager(window=10).summary()
    assert s.n == 0
    assert np.isnan(s.median_db)


def test_averager_window_clipping(carrier_iq):
    de = DifferentialEstimator()
    av = Averager(window=5)
    for _ in range(20):
        av.push(de.estimate(carrier_iq, 0.5 * carrier_iq))
    assert av.n == 5


def test_averager_median_matches_constant_input(carrier_iq):
    de = DifferentialEstimator()
    av = Averager(window=10)
    for _ in range(10):
        av.push(de.estimate(2.0 * carrier_iq, carrier_iq))
    s = av.summary()
    assert s.median_db == pytest.approx(6.020599, abs=1e-3)
    assert s.se_db == pytest.approx(0.0, abs=1e-9)


def test_averager_hac_se_shrinks_with_n():
    # synthetic Δ stream with iid noise — HAC SE should be ≈ σ/√n
    rng = np.random.default_rng(0)
    sigma = 1.0
    deltas_small = rng.standard_normal(20) * sigma
    deltas_large = rng.standard_normal(200) * sigma
    n = 50

    av_small = Averager(window=n, hac_lag=3)
    av_large = Averager(window=n, hac_lag=3)
    # Push raw values via a synthesised estimate object
    from sdr_kunchenko.doa.differential import DifferentialEstimate

    def _push(av: Averager, vals):
        for v in vals:
            est = DifferentialEstimate(
                delta_db=float(v), se_db=0.0, ci95_db=(float(v), float(v)),
                amp_a=1.0, amp_b=1.0, se_a=0.0, se_b=0.0, estimator_name="t",
            )
            av.push(est)

    _push(av_small, deltas_small)
    _push(av_large, deltas_large)
    s_small = av_small.summary()
    s_large = av_large.summary()
    # Both windows are clipped to `n=50`; large stream fills the window, small
    # one only partially → larger n inside window → smaller SE.
    assert s_large.se_db < s_small.se_db


def test_averager_per_hop_bucketing_groups_by_fc():
    de = DifferentialEstimator()
    iq1 = np.array([2.0 + 0j] * 8, dtype=np.complex128)
    iq2 = np.array([1.0 + 0j] * 8, dtype=np.complex128)
    av = Averager(window=20, bucket_by_hop=True, bucket_step_hz=1e6)
    # Two hops, each given many estimates with the same Δ
    for _ in range(8):
        av.push(de.estimate(iq1, iq2), fc_offset_hz=2_400_000.0)
        av.push(de.estimate(2 * iq1, iq2), fc_offset_hz=5_800_000.0)
    s = av.summary()
    assert s.n_buckets == 2
    # Per-bucket medians: 6.02 and 12.04, → median of medians ≈ 9.03 dB
    assert s.median_db == pytest.approx(9.03, abs=0.05)


def test_offline_summary_helper(carrier_iq):
    de = DifferentialEstimator()
    estimates = [de.estimate(2.0 * carrier_iq, carrier_iq) for _ in range(5)]
    s = offline_summary(estimates, window=10, hac_lag=2)
    assert s.median_db == pytest.approx(6.020599, abs=1e-3)


# ---------------------------------------------------------------- null detector


def _summary(median: float, se: float) -> AveragerSummary:
    z = 1.959964
    return AveragerSummary(
        n=10, median_db=median, se_db=se,
        ci95_db=(median - z * se, median + z * se),
        n_buckets=1, estimator_names=("t",),
    )


def test_null_detector_triggers_after_k_consecutive():
    nd = NullDetector(abs_db_thr=0.5, ci_width_thr=1.0, min_consecutive=3)
    s_in = _summary(0.1, 0.1)
    assert nd.update(s_in) is False
    assert nd.update(s_in) is False
    assert nd.update(s_in) is True
    # one bad sample resets
    assert nd.update(_summary(2.0, 0.1)) is False


def test_null_detector_ignores_wide_ci():
    nd = NullDetector(abs_db_thr=0.5, ci_width_thr=1.0, min_consecutive=2)
    # CI width = 2*z*5 ≈ 19.6 dB → above thr
    s = _summary(0.1, 5.0)
    assert nd.update(s) is False
    assert nd.update(s) is False
    assert nd.consecutive == 0


def test_null_detector_resets_on_nan():
    nd = NullDetector(min_consecutive=2)
    nd.update(_summary(0.1, 0.1))
    assert nd.update(_summary(float("nan"), float("nan"))) is False
    assert nd.consecutive == 0


# ---------------------------------------------------------------- inverse map


def test_inverse_map_monotonic_in_active_sector(wide_geom):
    grid, deltas = predicted_delta_db_grid(wide_geom)
    # restrict to the active overlap of the wide-beam config: |θ| ≤ ~30°
    mask = (grid >= -30.0) & (grid <= 30.0)
    diffs = np.diff(deltas[mask])
    assert (diffs >= 0).all() or (diffs <= 0).all()


@pytest.mark.parametrize("theta_true", [-60.0, -30.0, 0.0, 30.0, 60.0])
def test_inverse_map_round_trip(wide_geom, theta_true):
    """Round-trip θ → Δ → θ̂ on the same geometry."""
    _, deltas = predicted_delta_db_grid(
        wide_geom, theta_grid_deg=np.array([theta_true]),
    )
    theta_hat = theta_from_delta_db(float(deltas[0]), wide_geom)
    assert abs(theta_hat - theta_true) < 1.0


# ----------------------------------------------- calibration model-mismatch
# These guard the manuscript's central honesty point: the matched
# forward/inverse round trip is exact by construction (a consistency check),
# but a calibration mismatch makes the angle error real and conditioning-
# dependent — exactly what analyze_mismatch.py / analyze_conditioning.py report.


def _delta_true(theta_deg: float, geom: AntennaGeometry) -> float:
    g_a = cosine_beam_pattern(theta_deg, geom.boresight_a_deg,
                              geom.beamwidth_3db_deg, floor_db=geom.pattern_floor_db)
    g_b = cosine_beam_pattern(theta_deg, geom.boresight_b_deg,
                              geom.beamwidth_3db_deg, floor_db=geom.pattern_floor_db)
    return float(10.0 * np.log10(float(g_a) / float(g_b)))


def test_gain_imbalance_makes_error_real_at_crossover(wide_geom):
    """Matched inverse ≈ 0°; a 1 dB channel-gain imbalance gives a real ~2° error."""
    d0 = _delta_true(0.0, wide_geom)
    assert abs(theta_from_delta_db(d0, wide_geom)) < 0.2          # matched: tautological
    err = theta_from_delta_db(d0 + 1.0, wide_geom)               # +1 dB imbalance
    assert 1.0 < abs(err) < 5.0                                   # σ_θ(1 dB) ≈ 2.2° at θ=0


def test_mismatch_error_worse_in_saturated_sector(wide_geom):
    """The same 1 dB imbalance is worse in the beam-floored sector than at θ=0."""
    err0 = abs(theta_from_delta_db(_delta_true(0.0, wide_geom) + 1.0, wide_geom) - 0.0)
    err45 = abs(theta_from_delta_db(_delta_true(-45.0, wide_geom) + 1.0, wide_geom) - (-45.0))
    assert err45 > err0


def test_boresight_offset_biases_estimate(wide_geom):
    """A +2° array mispointing biases θ̂ by ≈ +2° in the well-conditioned sector."""
    assumed = dataclasses.replace(
        wide_geom, boresight_a_deg=wide_geom.boresight_a_deg + 2.0,
        boresight_b_deg=wide_geom.boresight_b_deg + 2.0)
    theta_hat = theta_from_delta_db(_delta_true(0.0, wide_geom), assumed)
    assert abs(theta_hat - 2.0) < 1.0


# --------------------------------------------------------- end-to-end (mini)


def test_end_to_end_small_acceptance(carrier_iq, wide_geom):
    """Mini A4 acceptance: 5 angles × 20 pairs at SNR 30 dB → θ̂ within ±5°."""
    de = DifferentialEstimator()
    fs = 10e6
    for theta in (-60.0, -30.0, 0.0, 30.0, 60.0):
        av = Averager(window=20, hac_lag=3)
        for k in range(20):
            res = simulate_dual_channel(
                carrier_iq, theta_target_deg=theta, fs=fs, geometry=wide_geom,
                snr_db=30.0, rng=np.random.default_rng(k),
            )
            av.push(de.estimate(res.iq_a, res.iq_b))
        s = av.summary()
        theta_hat = theta_from_delta_db(s.median_db, wide_geom)
        assert abs(theta_hat - theta) <= 5.0, (
            f"theta={theta} got θ̂={theta_hat:.2f} (Δ̄={s.median_db:.2f})"
        )
