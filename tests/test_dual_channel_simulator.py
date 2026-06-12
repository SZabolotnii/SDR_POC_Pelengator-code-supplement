"""Tests for the synthetic dual-antenna geometry simulator (Phase 7 / A2).

Acceptance from the plan:
    * θ_target = 0 → |Δ_dB| < 0.1 (perfect cross-over).
    * |θ_target| > 30 → |Δ_dB| > 3 (one antenna clearly louder).

We additionally pin the cosine² beam shape itself, the array-phase-only
behaviour for a wide-beam config, multipath superposition, and the
documented blind sector beyond ±BW₃ for both antennas (where Δ ≈ 0).
"""
from __future__ import annotations

import numpy as np
import pytest

from sdr_kunchenko.rf.dual_channel_simulator import (
    AntennaGeometry,
    MultipathTap,
    cosine_beam_pattern,
    simulate_dual_channel,
)

# ---------------------------------------------------------------- helpers


def _delta_db(iq_a: np.ndarray, iq_b: np.ndarray) -> float:
    pa = float(np.mean(np.abs(iq_a) ** 2))
    pb = float(np.mean(np.abs(iq_b) ** 2))
    return 10.0 * np.log10(pa / pb)


@pytest.fixture
def carrier_iq() -> np.ndarray:
    """4096-sample CW (unit-amplitude complex tone)."""
    n = 4096
    fs = 10e6
    f0 = 250e3
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * f0 * t).astype(np.complex128)


# ------------------------------------------------------- beam-pattern shape


def test_beam_pattern_peak_minus3db_floor():
    """G(boresight) = 1, G(±BW₃/2) = 0.5, G beyond BW₃ floored."""
    g0 = cosine_beam_pattern(0.0, boresight_deg=0.0, beamwidth_3db_deg=30.0)
    assert g0 == pytest.approx(1.0)
    g_minus3 = cosine_beam_pattern(15.0, boresight_deg=0.0, beamwidth_3db_deg=30.0)
    assert 10.0 * np.log10(g_minus3) == pytest.approx(-3.01, abs=0.05)
    g_far = cosine_beam_pattern(180.0, boresight_deg=0.0, beamwidth_3db_deg=30.0,
                                  floor_db=-50.0)
    assert 10.0 * np.log10(g_far) == pytest.approx(-50.0, abs=0.1)


# ----------------------------------------------------- on-axis cross-over


def test_zero_angle_delta_below_0p1_db(carrier_iq):
    res = simulate_dual_channel(
        carrier_iq, theta_target_deg=0.0, fs=10e6,
        snr_db=np.inf, rng=np.random.default_rng(0),
    )
    assert abs(_delta_db(res.iq_a, res.iq_b)) < 0.1
    # both antennas see −3 dB on boresight cross-over → equal power
    assert res.info["g_a_main"] == pytest.approx(0.5, rel=1e-3)
    assert res.info["g_b_main"] == pytest.approx(0.5, rel=1e-3)


# --------------------------------------------------- off-boresight imbalance


@pytest.mark.parametrize("theta_deg", [-40.0, -35.0, -31.0, 31.0, 35.0, 40.0])
def test_offboresight_imbalance(carrier_iq, theta_deg):
    """Within ±BW₃ of one antenna's boresight (±BW₃ ≈ ±30° from the array
    centre) the louder antenna is the one whose boresight is on the same
    side as θ. Beyond ±BW₃+|φ| (≈ ±45°) both antennas saturate at the floor
    and Δ ≈ 0 — that range is exercised in `test_blind_sector_beyond_main_lobes`.
    """
    res = simulate_dual_channel(
        carrier_iq, theta_target_deg=theta_deg, fs=10e6,
        snr_db=np.inf, rng=np.random.default_rng(0),
    )
    delta = _delta_db(res.iq_a, res.iq_b)
    assert abs(delta) > 3.0, f"theta={theta_deg}: |Δ|={abs(delta):.2f} dB"
    # sign matches geometry: target on A's side (θ > 0) → A wins (Δ_dB > 0).
    # Use `==` not `is`: numpy.bool_ does not satisfy `is True`.
    assert bool(delta > 0) == (theta_deg > 0)


# --------------------------------------------------------- monotonic sector


def test_monotonic_inside_active_overlap(carrier_iq):
    """Inside the overlap zone where *both* antennas are above the floor —
    i.e. |θ| < (BW₃/2 + |φ|) − BW₃/2 = |φ| with default geom (φ=±15°, BW=30°)
    — Δ_dB increases monotonically with θ. Beyond that one antenna floors and
    the trend can reverse: that's the documented edge of the usable sector.
    """
    deltas = []
    for theta in np.arange(-12.0, 12.001, 3.0):
        res = simulate_dual_channel(
            carrier_iq, theta_target_deg=float(theta), fs=10e6,
            snr_db=np.inf, rng=np.random.default_rng(0),
        )
        deltas.append(_delta_db(res.iq_a, res.iq_b))
    diffs = np.diff(deltas)
    assert np.all(diffs > 0), f"Δ not monotonic in active overlap: {deltas}"


# ------------------------------------------------------- array-phase-only


def test_array_phase_only_with_omni_beam(carrier_iq):
    """Wide beam (BW=180°) → both gains ≈ 1; only array phase distinguishes."""
    geom = AntennaGeometry(
        boresight_a_deg=0.0, boresight_b_deg=0.0,
        beamwidth_3db_deg=180.0, spacing_m=0.0625, fc_hz=2.412e9,
    )
    res = simulate_dual_channel(
        carrier_iq, theta_target_deg=30.0, fs=10e6, geometry=geom,
        snr_db=np.inf, rng=np.random.default_rng(0),
    )
    # equal magnitude
    assert _delta_db(res.iq_a, res.iq_b) == pytest.approx(0.0, abs=0.05)
    # phase between channels matches expected π (d/λ) sin(θ) — between B and A.
    lam = 299_792_458.0 / 2.412e9
    expected_phase = 2.0 * np.pi * (geom.spacing_m / lam) * np.sin(np.deg2rad(30.0))
    rel = res.iq_b * np.conj(res.iq_a)
    measured_phase = float(np.angle(rel.mean()))
    # wrap to (-π, π]
    err = (measured_phase - expected_phase + np.pi) % (2 * np.pi) - np.pi
    assert abs(err) < 1e-3


# ----------------------------------------------------- AWGN matches request


def test_awgn_snr_close_to_target(carrier_iq):
    snr_db = 20.0
    res = simulate_dual_channel(
        carrier_iq, theta_target_deg=0.0, fs=10e6, snr_db=snr_db,
        rng=np.random.default_rng(123),
    )
    # carrier on boresight cross-over → power ≈ 0.5 |iq|², plus σ²
    sig_pow = 0.5 * float(np.mean(np.abs(carrier_iq) ** 2))
    n_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    measured_pow = float(np.mean(np.abs(res.iq_a) ** 2))
    assert measured_pow == pytest.approx(sig_pow + n_pow, rel=0.15)


# ----------------------------------------------------------- multipath


def test_multipath_changes_output(carrier_iq):
    """Adding a tap with non-zero gain must change at least one channel."""
    base = simulate_dual_channel(
        carrier_iq, theta_target_deg=20.0, fs=10e6, snr_db=np.inf,
        rng=np.random.default_rng(0),
    )
    with_tap = simulate_dual_channel(
        carrier_iq, theta_target_deg=20.0, fs=10e6, snr_db=np.inf,
        multipath_taps=[
            MultipathTap(delay_s=0.5e-6, gain=0.4 + 0.0j, theta_deg=-40.0),
        ],
        rng=np.random.default_rng(0),
    )
    # noise-free; any difference must come from the tap
    assert not np.allclose(base.iq_a, with_tap.iq_a)
    assert not np.allclose(base.iq_b, with_tap.iq_b)
    assert with_tap.info["n_multipath_taps"] == 1


# ----------------------------------------------------- documented blind sector


def test_blind_sector_beyond_main_lobes(carrier_iq):
    """|θ| ≥ BW₃ from both boresights → both gains floored, Δ ≈ 0."""
    res = simulate_dual_channel(
        carrier_iq, theta_target_deg=70.0, fs=10e6, snr_db=np.inf,
        rng=np.random.default_rng(0),
    )
    assert abs(_delta_db(res.iq_a, res.iq_b)) < 0.5


# ------------------------------------------------------ noise-free determinism


def test_deterministic_with_same_seed(carrier_iq):
    res1 = simulate_dual_channel(
        carrier_iq, theta_target_deg=10.0, fs=10e6, snr_db=15.0,
        rng=np.random.default_rng(42),
    )
    res2 = simulate_dual_channel(
        carrier_iq, theta_target_deg=10.0, fs=10e6, snr_db=15.0,
        rng=np.random.default_rng(42),
    )
    assert np.allclose(res1.iq_a, res2.iq_a)
    assert np.allclose(res1.iq_b, res2.iq_b)


# ------------------------------------------------------- input validation


def test_rejects_2d_iq():
    iq2d = np.zeros((10, 10), dtype=np.complex128)
    with pytest.raises(ValueError):
        simulate_dual_channel(iq2d, theta_target_deg=0.0, fs=10e6)
