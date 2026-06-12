"""Streaming OcuSync-style burst detector for raw complex IQ.

Energy-envelope Schmitt-trigger pipeline:
    1. instantaneous power p[t] = 10·log10(|iq[t]|²)
    2. p̄[t] = rolling mean over `smoothing_us` µs window
    3. noise floor = quantile(p̄, `noise_quantile`)
    4. on/off via two-level hysteresis (high = noise+threshold, low = high − hysteresis)
    5. drop bursts shorter than `min_duration_us` µs
    6. fc_offset_hz = peak STFT bin inside the burst (one-sided abs)

GSA-CUSUM (`sdr_kunchenko.gsa.GSACUSUMDetector`) operates on a *real-valued*
change-point statistic and fires at the first alarm only — useful for
change-point detection inside a known transmission, but not directly for
locating multiple OcuSync bursts in a long IQ stream. We therefore use the
faster envelope detector here and reserve GSA-CUSUM for downstream
within-burst event detection.

Tampere I/Q (Karel Pärlin format): little-endian int16 interleaved I,Q at
fs = 120 MS/s for 2.4 GHz captures (200 MS/s for 5.8 GHz). Use
`load_tampere_iq` to memory-map a chunk as complex128.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import stft


# --------------------------------------------------------------- data classes


@dataclass(frozen=True)
class Burst:
    """One detected OcuSync (or OcuSync-like) packet.

    Sample indices are inclusive-start, exclusive-end (Python slice semantics).
    `fc_offset_hz` is signed: positive means above the IQ centre frequency.
    """
    t_start: int
    t_end: int
    fc_offset_hz: float
    peak_power_db: float

    @property
    def duration_samples(self) -> int:
        return self.t_end - self.t_start

    def duration_seconds(self, fs: float) -> float:
        return self.duration_samples / fs

    def to_dict(self) -> dict:
        return {
            "t_start": int(self.t_start),
            "t_end": int(self.t_end),
            "fc_offset_hz": float(self.fc_offset_hz),
            "peak_power_db": float(self.peak_power_db),
        }


# --------------------------------------------------------------------- helpers


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Same-length rolling mean with edge replication."""
    if window <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    pad = window // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def _schmitt_trigger(envelope: np.ndarray, *, high: float, low: float) -> list[tuple[int, int]]:
    """Return list of (start, end) intervals where the envelope was on.

    State flips on at envelope ≥ high, flips off at envelope < low.
    Intervals follow Python slice semantics: end is exclusive.
    """
    intervals: list[tuple[int, int]] = []
    state = False
    cur_start = 0
    for t in range(len(envelope)):
        v = envelope[t]
        if not state and v >= high:
            state = True
            cur_start = t
        elif state and v < low:
            state = False
            intervals.append((cur_start, t))
    if state:
        intervals.append((cur_start, len(envelope)))
    return intervals


def _peak_freq_in_burst(iq_seg: np.ndarray, fs: float, nfft: int) -> tuple[float, float]:
    """Average PSD over the burst, return (fc_offset_hz, peak_power_db)."""
    seg = np.asarray(iq_seg)
    nperseg = min(nfft, len(seg))
    if nperseg < 8:
        return 0.0, 10.0 * np.log10(np.mean(np.abs(seg) ** 2) + 1e-30)
    # two-sided spectrum so positive AND negative offsets are recoverable
    f, _, Z = stft(
        seg, fs=fs, window="blackman", nperseg=nperseg, noverlap=nperseg // 2,
        nfft=nfft, return_onesided=False, boundary=None, padded=False,
    )
    psd = (np.abs(Z) ** 2).mean(axis=1)
    peak_idx = int(np.argmax(psd))
    fc_offset_hz = float(f[peak_idx])
    peak_power_db = 10.0 * np.log10(psd[peak_idx] + 1e-30)
    return fc_offset_hz, peak_power_db


# ---------------------------------------------------------------- core API


def detect_bursts(
    iq: np.ndarray,
    fs: float,
    *,
    smoothing_us: float = 1.0,
    threshold_db: float = 10.0,
    hysteresis_db: float = 3.0,
    min_duration_us: float = 5.0,
    noise_quantile: float = 0.10,
    stft_nfft: int = 512,
) -> list[Burst]:
    """Detect every burst in a complex IQ chunk.

    Parameters
    ----------
    iq : (N,) complex array
        Baseband samples.
    fs : float
        Sample rate in Hz.
    smoothing_us : float
        Rolling-mean window for the log-power envelope (µs). Should be much
        shorter than the shortest burst (default 1 µs ≈ 120 samples @ 120 MS/s).
    threshold_db : float
        Schmitt-high level above the empirical noise floor.
    hysteresis_db : float
        Schmitt-low level is `threshold_db − hysteresis_db` above the floor.
    min_duration_us : float
        Discard bursts shorter than this (filters out STFT artefacts and
        impulsive interferers).
    noise_quantile : float
        Quantile of the smoothed envelope used as the noise-floor estimate.
        0.10 is robust when bursts cover ≤80 % of the stream.
    stft_nfft : int
        FFT size for the within-burst peak-frequency estimate.

    Returns
    -------
    list of `Burst` (sorted by `t_start`).
    """
    iq = np.asarray(iq).ravel()
    if not np.iscomplexobj(iq):
        raise TypeError(f"Expected complex IQ, got dtype {iq.dtype}")
    if iq.size == 0:
        return []

    eps = 1e-30
    p_db = 10.0 * np.log10(np.abs(iq) ** 2 + eps)

    smoothing_n = max(1, int(round(smoothing_us * 1e-6 * fs)))
    p_smooth = _moving_average(p_db, smoothing_n)

    noise_floor = float(np.quantile(p_smooth, noise_quantile))
    high = noise_floor + threshold_db
    low = high - hysteresis_db

    raw_intervals = _schmitt_trigger(p_smooth, high=high, low=low)
    min_samples = max(1, int(round(min_duration_us * 1e-6 * fs)))

    bursts: list[Burst] = []
    for s, e in raw_intervals:
        if (e - s) < min_samples:
            continue
        fc_offset_hz, peak_power_db = _peak_freq_in_burst(iq[s:e], fs, stft_nfft)
        bursts.append(
            Burst(t_start=s, t_end=e, fc_offset_hz=fc_offset_hz,
                   peak_power_db=peak_power_db)
        )
    return bursts


def iter_bursts_chunked(
    iq_source: np.ndarray | "np.memmap",
    fs: float,
    *,
    chunk_seconds: float = 0.1,
    overlap_us: float = 50.0,
    **detect_kwargs,
) -> Iterator[Burst]:
    """Stream bursts from a long memory-mapped IQ array in chunks.

    Adjacent chunks overlap by `overlap_us` so a burst spanning a chunk
    boundary is recovered from the chunk that contains its centre. Bursts
    whose centre lies inside the overlap region are de-duplicated by start
    sample (with an epsilon of one smoothing window).
    """
    n_total = int(np.asarray(iq_source).size)
    chunk_n = max(1, int(round(chunk_seconds * fs)))
    overlap_n = max(0, int(round(overlap_us * 1e-6 * fs)))
    smoothing_n = max(1, int(round(detect_kwargs.get("smoothing_us", 1.0) * 1e-6 * fs)))

    seen_starts: list[int] = []
    pos = 0
    while pos < n_total:
        end = min(n_total, pos + chunk_n)
        chunk = np.asarray(iq_source[pos:end])
        if not np.iscomplexobj(chunk):
            chunk = chunk.astype(np.complex128, copy=False)
        local_bursts = detect_bursts(chunk, fs, **detect_kwargs)
        for b in local_bursts:
            global_start = pos + b.t_start
            global_end = pos + b.t_end
            mid = (global_start + global_end) // 2
            # Skip bursts whose centre is in the leading overlap region
            # (they were already emitted by the previous chunk).
            if pos > 0 and mid < pos + overlap_n:
                continue
            # Dedup by proximity of start to a previously-emitted start.
            if any(abs(global_start - s) < smoothing_n for s in seen_starts):
                continue
            seen_starts.append(global_start)
            yield Burst(
                t_start=global_start, t_end=global_end,
                fc_offset_hz=b.fc_offset_hz, peak_power_db=b.peak_power_db,
            )
        if end == n_total:
            break
        pos = end - overlap_n


# ---------------------------------------------------------------- Tampere I/O


def load_tampere_iq(
    path: str | Path,
    *,
    n_samples: int | None = None,
    offset_samples: int = 0,
) -> np.ndarray:
    """Load Karel Pärlin Tampere int16 interleaved I/Q as complex128.

    Each complex sample is one (I, Q) pair of little-endian int16. The result
    is a contiguous `complex128` array of length up to `n_samples` (or the
    whole file if `n_samples is None`).

    Parameters
    ----------
    path : path to .bin file (e.g. data/tampere_extract/DJI_mavic_pro_2G.bin)
    n_samples : number of complex samples to load (None → all)
    offset_samples : skip this many complex samples from the start
    """
    raw = np.fromfile(
        path,
        dtype="<i2",
        count=-1 if n_samples is None else 2 * n_samples,
        offset=2 * 2 * offset_samples,  # 2 int16 per complex × 2 bytes per int16
    )
    n_complex = raw.size // 2
    raw = raw[: 2 * n_complex]
    return raw.astype(np.float64).view(np.complex128)


def memmap_tampere_iq(path: str | Path) -> np.memmap:
    """Memory-map a Tampere bin file as int16 LE — caller views I/Q pairs.

    Returns a flat int16 memmap of length 2·N (interleaved). Build a complex
    chunk from samples [2*i, 2*i+1] as I + 1j·Q. Useful when the file is
    larger than RAM and we want lazy chunked reading via
    `iter_bursts_chunked`.
    """
    return np.memmap(path, dtype="<i2", mode="r")


def tampere_chunk_to_complex(chunk_int16: np.ndarray) -> np.ndarray:
    """Convert a (2·N,) int16 view into an (N,) complex128 array."""
    n_complex = chunk_int16.size // 2
    return (
        chunk_int16[: 2 * n_complex].astype(np.float64).view(np.complex128).copy()
    )


__all__ = [
    "Burst",
    "detect_bursts",
    "iter_bursts_chunked",
    "load_tampere_iq",
    "memmap_tampere_iq",
    "tampere_chunk_to_complex",
]
