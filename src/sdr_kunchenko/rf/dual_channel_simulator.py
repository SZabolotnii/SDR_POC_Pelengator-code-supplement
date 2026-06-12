"""Synthetic dual-antenna geometry over a single-channel IQ recording.

Phase 7 / A2 — turns a Tampere single-channel OcuSync capture into a
synthetic (iq_a, iq_b) pair as if it had been received by two
boresight-tilted antennas with a known angular target. Used by the
proxy-pipeline (A5) to test the differential-amplitude estimator (A4)
without live SDR hardware.

Model per arrival path (main path + each multipath tap):
    iq_a(θ) = √G_a(θ) · gain · iq · exp(-j π (d/λ) sin θ) · z⁻ᵏ
    iq_b(θ) = √G_b(θ) · gain · iq · exp(+j π (d/λ) sin θ) · z⁻ᵏ
where:
    G_a(θ) = max( cos²((θ - φ_a) / (BW₃/2) · π/4), ε )    cos² beam
    G_b(θ) = max( cos²((θ - φ_b) / (BW₃/2) · π/4), ε )
    φ_a, φ_b — boresights of antennas A and B (deg)
    BW₃     — full-3-dB beamwidth of each antenna (deg)
    d       — element spacing (m)
    λ       — c / fc
    k       — delay in samples

Beyond ±BW₃ the cos² argument is clipped to ±π/2, so the antenna gain
saturates at the floor ε (default −60 dB). With the default geometry
(φ = ±15°, BW₃ = 30°) the sector |θ| > ~45° is therefore a blind zone
where both antennas are floored and Δ ≈ 0; this matches a real two-patch
front-end and is documented as a known limitation. Per-channel
zero-mean complex AWGN is added at the requested SNR after the path sum.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

C0 = 299_792_458.0  # speed of light, m/s


# ---------------------------------------------------------------- types


@dataclass(frozen=True)
class AntennaGeometry:
    """Two-antenna front-end with cos² beams + half-wavelength array spacing.

    Defaults reproduce the planned −3 dB cross-over at θ=0 and a
    full 3-dB beamwidth of 30° per antenna at fc = 2.412 GHz.
    """
    boresight_a_deg: float = +15.0
    boresight_b_deg: float = -15.0
    beamwidth_3db_deg: float = 30.0
    spacing_m: float = 0.0625      # ~λ/2 at 2.4 GHz
    fc_hz: float = 2.412e9
    pattern_floor_db: float = -60.0


@dataclass(frozen=True)
class MultipathTap:
    """One reflected/diffracted copy of the signal."""
    delay_s: float
    gain: complex
    theta_deg: float


@dataclass(frozen=True)
class SimulationResult:
    """Return container with both channels and a small diagnostics dict."""
    iq_a: np.ndarray
    iq_b: np.ndarray
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- DSP


def cosine_beam_pattern(
    theta_deg: float | np.ndarray,
    boresight_deg: float,
    beamwidth_3db_deg: float,
    *,
    floor_db: float = -60.0,
) -> np.ndarray:
    """Linear (power-domain) cos² antenna gain.

    G(θ) = cos²((θ-φ)/(BW₃/2) · π/4), clamped above `floor_db`.
    G(boresight) = 1; G(boresight ± BW₃/2) = 0.5 (=−3 dB);
    G(boresight ± BW₃)   = 0  → floored.
    """
    diff = np.deg2rad(np.asarray(theta_deg) - boresight_deg)
    half = np.deg2rad(beamwidth_3db_deg / 2.0)
    arg = np.clip(diff / half * (np.pi / 4.0), -np.pi / 2, np.pi / 2)
    g = np.cos(arg) ** 2
    floor = 10.0 ** (floor_db / 10.0)
    return np.maximum(g, floor)


def _project_path(
    iq: np.ndarray, theta_deg: float, geom: AntennaGeometry, gain: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-antenna √G gain + ±half-array array-phase to one path."""
    g_a = cosine_beam_pattern(
        theta_deg, geom.boresight_a_deg, geom.beamwidth_3db_deg,
        floor_db=geom.pattern_floor_db,
    )
    g_b = cosine_beam_pattern(
        theta_deg, geom.boresight_b_deg, geom.beamwidth_3db_deg,
        floor_db=geom.pattern_floor_db,
    )
    lam = C0 / geom.fc_hz
    phi = np.pi * (geom.spacing_m / lam) * np.sin(np.deg2rad(theta_deg))
    iq_a = np.sqrt(float(g_a)) * iq * np.exp(-1j * phi) * gain
    iq_b = np.sqrt(float(g_b)) * iq * np.exp(+1j * phi) * gain
    return iq_a, iq_b


def _delay_samples(iq: np.ndarray, delay_n: int) -> np.ndarray:
    """Causal integer-sample delay (zero-fill on the left)."""
    if delay_n <= 0:
        return iq.copy()
    out = np.zeros_like(iq)
    out[delay_n:] = iq[: iq.size - delay_n]
    return out


def _add_awgn(
    iq: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean circularly symmetric complex Gaussian noise.

    SNR is signal-power (mean |iq|²) over noise-power per channel. If the
    input has zero power we simply return it unchanged — adding noise to a
    silent channel would deliver an undefined SNR.
    """
    sig_pow = float(np.mean(np.abs(iq) ** 2))
    if sig_pow <= 0.0 or not np.isfinite(snr_db):
        return iq.copy()
    snr_lin = 10.0 ** (snr_db / 10.0)
    n_pow = sig_pow / snr_lin
    sigma = np.sqrt(n_pow / 2.0)
    n = rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)
    return iq + sigma * n


# ---------------------------------------------------------------- public API


def simulate_dual_channel(
    iq_single: np.ndarray,
    *,
    theta_target_deg: float,
    fs: float,
    geometry: AntennaGeometry | None = None,
    snr_db: float = 40.0,
    multipath_taps: Sequence[MultipathTap] | None = None,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Render a single-channel baseband recording into a dual-antenna pair.

    Parameters
    ----------
    iq_single : (N,) complex array
        Source baseband samples (one antenna).
    theta_target_deg : float
        Direction of arrival of the dominant path, in degrees.
    fs : float
        Sample rate in Hz (used to convert multipath delays to samples).
    geometry : AntennaGeometry, optional
        Front-end model. Defaults give ±15° boresights, BW₃ = 30°, λ/2
        spacing at 2.412 GHz.
    snr_db : float
        SNR of each output channel after gain (signal-to-noise per channel).
        Use ``np.inf`` for noise-free output.
    multipath_taps : sequence of MultipathTap, optional
        Each tap contributes a delayed, scaled copy of the signal arriving
        from its own θ. Useful for stress-testing A4 averaging.
    rng : np.random.Generator, optional
        For reproducibility. Defaults to a fresh PCG64.
    """
    iq = np.ascontiguousarray(iq_single, dtype=np.complex128)
    if iq.ndim != 1:
        raise ValueError(f"expected 1-D iq_single, got shape {iq.shape}")
    geom = geometry or AntennaGeometry()
    rng = rng or np.random.default_rng()

    iq_a, iq_b = _project_path(iq, theta_target_deg, geom, gain=1.0 + 0.0j)

    n_taps = 0
    if multipath_taps:
        for tap in multipath_taps:
            delay_n = int(round(tap.delay_s * fs))
            iq_d = _delay_samples(iq, delay_n)
            d_a, d_b = _project_path(iq_d, tap.theta_deg, geom, gain=tap.gain)
            iq_a = iq_a + d_a
            iq_b = iq_b + d_b
            n_taps += 1

    iq_a = _add_awgn(iq_a, snr_db, rng)
    iq_b = _add_awgn(iq_b, snr_db, rng)

    g_a = cosine_beam_pattern(
        theta_target_deg, geom.boresight_a_deg, geom.beamwidth_3db_deg,
        floor_db=geom.pattern_floor_db,
    )
    g_b = cosine_beam_pattern(
        theta_target_deg, geom.boresight_b_deg, geom.beamwidth_3db_deg,
        floor_db=geom.pattern_floor_db,
    )
    info = {
        "theta_target_deg": float(theta_target_deg),
        "fs": float(fs), "snr_db": float(snr_db),
        "g_a_main": float(g_a), "g_b_main": float(g_b),
        "expected_delta_db_main": float(10.0 * np.log10(g_a / g_b)),
        "n_multipath_taps": n_taps,
        "geometry": geom,
    }
    return SimulationResult(iq_a=iq_a, iq_b=iq_b, info=info)


__all__ = [
    "AntennaGeometry",
    "MultipathTap",
    "SimulationResult",
    "cosine_beam_pattern",
    "simulate_dual_channel",
]
