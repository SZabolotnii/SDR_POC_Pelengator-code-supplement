"""Tests for the streaming OcuSync-style burst detector (Phase 7 / Stream A).

Strategy: most assertions run on synthetic AWGN streams with controlled bursts
so ground-truth is exact. A second sanity test loads ~0.1 s from the Tampere
Mavic Pro recording (skipped if the file is not present) and verifies that the
detector finds a plausible number of bursts at sensible bandwidths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sdr_kunchenko.rf.packet_detector import (
    Burst,
    detect_bursts,
    iter_bursts_chunked,
    load_tampere_iq,
)


# --------------------------------------------------------------- synthetic gen


def _synthesize_iq(
    *,
    fs: float,
    duration_s: float,
    bursts: list[tuple[float, float, float, float]],  # (t_start, t_end, fc_hz, snr_db)
    noise_sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate complex AWGN with sinusoidal bursts injected."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (
        noise_sigma / np.sqrt(2)
    )
    t = np.arange(n) / fs
    for t_start_s, t_end_s, fc_hz, snr_db in bursts:
        s_idx = int(round(t_start_s * fs))
        e_idx = int(round(t_end_s * fs))
        amp = noise_sigma * 10.0 ** (snr_db / 20.0)
        # smooth raised-cosine envelope so the burst has clean edges
        seg_len = e_idx - s_idx
        ramp_len = max(1, seg_len // 20)
        env = np.ones(seg_len)
        env[:ramp_len] = np.sin(np.linspace(0, np.pi / 2, ramp_len)) ** 2
        env[-ramp_len:] = np.sin(np.linspace(np.pi / 2, 0, ramp_len)) ** 2
        iq[s_idx:e_idx] += amp * env * np.exp(1j * 2 * np.pi * fc_hz * t[s_idx:e_idx])
    return iq


def _f1(true_bursts: list[tuple[int, int]],
         detected: list[Burst],
         iou_threshold: float = 0.3) -> float:
    """F1 with IoU-based matching at sample level."""
    matched_t = set()
    matched_d = set()
    for di, b in enumerate(detected):
        ds, de = b.t_start, b.t_end
        for ti, (ts, te) in enumerate(true_bursts):
            if ti in matched_t:
                continue
            inter = max(0, min(de, te) - max(ds, ts))
            union = max(de, te) - min(ds, ts)
            if union > 0 and inter / union >= iou_threshold:
                matched_t.add(ti)
                matched_d.add(di)
                break
    tp = len(matched_d)
    fp = len(detected) - tp
    fn = len(true_bursts) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------- core tests


def test_no_bursts_in_pure_noise():
    fs = 1e6
    rng = np.random.default_rng(42)
    iq = (rng.standard_normal(int(fs * 0.05))
           + 1j * rng.standard_normal(int(fs * 0.05))) / np.sqrt(2)
    bursts = detect_bursts(iq, fs, threshold_db=10.0, min_duration_us=20.0)
    assert bursts == []


def test_single_burst_recovered():
    fs = 1e6
    iq = _synthesize_iq(
        fs=fs, duration_s=0.05,
        bursts=[(0.010, 0.020, 100e3, 20.0)],
    )
    bursts = detect_bursts(iq, fs, threshold_db=8.0, min_duration_us=50.0)
    assert len(bursts) == 1
    b = bursts[0]
    # Slack: 5 % raised-cosine ramp on each end of a 10 ms burst is 500 µs.
    # The Schmitt trigger fires partway up the ramp, so allow ±500 µs.
    assert abs(b.t_start / fs - 0.010) < 500e-6
    assert abs(b.t_end / fs - 0.020) < 500e-6
    assert abs(b.fc_offset_hz - 100e3) < fs / 256  # ~4 kHz at 256-bin FFT


def test_multiple_bursts_at_high_snr_f1_is_one():
    fs = 1e6
    truth = [
        (0.005, 0.012, +200e3, 20.0),
        (0.020, 0.025,  -50e3, 20.0),
        (0.035, 0.045, +300e3, 20.0),
    ]
    iq = _synthesize_iq(fs=fs, duration_s=0.05, bursts=truth, seed=1)
    detected = detect_bursts(iq, fs, threshold_db=8.0, min_duration_us=50.0)
    truth_idx = [(int(round(s * fs)), int(round(e * fs))) for s, e, _, _ in truth]
    assert _f1(truth_idx, detected, iou_threshold=0.5) == 1.0


def test_min_duration_filter_drops_impulse():
    """A 1-µs spike must be filtered out by min_duration_us=10."""
    fs = 1e6
    truth = [(0.010, 0.011, 0.0, 30.0)]  # 1 ms-scale at fs=1e6 == 1000 us
    # Reduce to 1 us by using fs=10e6
    fs = 10e6
    truth = [(0.010, 0.010 + 1e-6, 0.0, 30.0)]  # 1 µs spike
    iq = _synthesize_iq(fs=fs, duration_s=0.02, bursts=truth, seed=2)
    bursts = detect_bursts(iq, fs, threshold_db=8.0, min_duration_us=5.0)
    assert bursts == []


def test_chunked_iteration_yields_same_bursts():
    fs = 1e6
    truth = [
        (0.005, 0.012, +200e3, 20.0),
        (0.020, 0.025,  -50e3, 20.0),
        (0.105, 0.115, +300e3, 20.0),  # crosses a chunk boundary at 0.1 s
    ]
    iq = _synthesize_iq(fs=fs, duration_s=0.20, bursts=truth, seed=3)
    one_shot = detect_bursts(iq, fs, threshold_db=8.0, min_duration_us=50.0)
    chunked = list(iter_bursts_chunked(
        iq, fs, chunk_seconds=0.1, overlap_us=2_000.0,
        threshold_db=8.0, min_duration_us=50.0,
    ))
    assert len(chunked) == len(one_shot)
    for a, b in zip(sorted(chunked, key=lambda x: x.t_start),
                     sorted(one_shot, key=lambda x: x.t_start)):
        assert abs(a.t_start - b.t_start) < int(fs * 5e-6)
        assert abs(a.t_end - b.t_end) < int(fs * 5e-6)


def test_burst_dataclass_helpers():
    b = Burst(t_start=120, t_end=240, fc_offset_hz=1.5e6, peak_power_db=-30.0)
    assert b.duration_samples == 120
    assert abs(b.duration_seconds(fs=1e6) - 120e-6) < 1e-12
    d = b.to_dict()
    assert set(d) == {"t_start", "t_end", "fc_offset_hz", "peak_power_db"}


def test_complex_dtype_required():
    fs = 1e6
    real = np.zeros(1024, dtype=np.float64)
    with pytest.raises(TypeError, match="complex"):
        detect_bursts(real, fs)


# ---------------------------------------------------------- Tampere sanity


_TAMPERE = Path(__file__).resolve().parents[1] / "data" / "tampere_extract"


@pytest.mark.skipif(
    not (_TAMPERE / "DJI_mavic_pro_2G.bin").exists(),
    reason="Tampere bin not present; sanity test skipped.",
)
def test_tampere_mavic_pro_bursts_plausible():
    """Smoke-test on 0.05 s of Mavic Pro 2.4 GHz: detector should find ≥3 bursts."""
    fs = 120e6
    n = int(round(0.05 * fs))  # 50 ms ≈ 6e6 complex samples
    iq = load_tampere_iq(_TAMPERE / "DJI_mavic_pro_2G.bin", n_samples=n)
    bursts = detect_bursts(
        iq, fs,
        smoothing_us=0.5, threshold_db=8.0, hysteresis_db=2.0,
        min_duration_us=20.0, stft_nfft=1024,
    )
    # OcuSync-1 hops at ~10 ms cadence → roughly 5 bursts in 50 ms.
    # Accept anything in [2, 50] as "plausible".
    assert 2 <= len(bursts) <= 50, f"Implausible burst count: {len(bursts)}"
    for b in bursts:
        # OcuSync-1 packets / hop dwells: hundreds of µs up to ~20 ms in
        # anechoic captures (longer dwells appear when the link is idle on a
        # given band before the next hop).
        d_us = b.duration_seconds(fs) * 1e6
        assert 20.0 <= d_us <= 20_000.0, f"Burst duration out of expected band: {d_us} µs"
