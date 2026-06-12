#!/usr/bin/env python3
"""Multipath robustness sweep for the dual-channel DOA proxy (A5 follow-up).

Closes the "multipath absent in A5" limitation. The fingerprint gate is run
once on the real Tampere bursts and the kept target bursts are cached; each
cached burst is then re-rendered through the dual-channel simulator with
multipath taps populated, across a sweep of reflected/scattered power, and the
final inverted angle is compared with the target. Two channel models:

  * 2-ray  — one specular reflection from a fixed off-axis wall;
  * Rayleigh — several diffuse taps with complex-Gaussian gains and random
               arrival angles (rich scattering).

The inverse uses the *true* geometry, so the only error source here is the
multipath channel (kept separate from the calibration-mismatch study).

Outputs:
  experiments/poc_pelengator/results/multipath_sensitivity.json
  manuscript/hait_poc_pelengator/figures/multipath_sensitivity.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sdr_kunchenko.doa import Averager, DifferentialEstimator, theta_from_delta_db  # noqa: E402
from sdr_kunchenko.rf.dual_channel_simulator import (  # noqa: E402
    AntennaGeometry,
    MultipathTap,
    simulate_dual_channel,
)
from sdr_kunchenko.rf.fingerprint import FingerprintFilter  # noqa: E402
from sdr_kunchenko.rf.packet_detector import detect_bursts  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG = HERE / "config.yaml"
RESULTS = HERE / "results" / "multipath_sensitivity.json"
FIGURE = REPO_ROOT / "manuscript" / "hait_poc_pelengator" / "figures" / "multipath_sensitivity.png"

# Relative reflected/scattered power sweep (dB rel. to direct path). None = clean.
REL_POWER_DB = [None, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0]
TWO_RAY_REFL_DEG = 50.0          # specular wall, off the well-conditioned sector
TWO_RAY_DELAY_S = 100e-9         # 100 ns -> 12 samples @ 120 MS/s
RAYLEIGH_N_TAPS = 6
RAYLEIGH_MAX_DELAY_S = 300e-9
TAP_SEED = 7                     # fixed taps across bursts & angles


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_iq_complex64(path: Path, duration_s: float, fs: float) -> np.ndarray:
    n_complex = int(round(duration_s * fs))
    data = np.fromfile(path, dtype="<i2", count=2 * n_complex)
    iq = data.astype(np.float32).reshape(-1, 2)
    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def build_geometry(cfg: dict) -> AntennaGeometry:
    g = cfg["geometry"]
    return AntennaGeometry(
        boresight_a_deg=float(g["boresight_a_deg"]),
        boresight_b_deg=float(g["boresight_b_deg"]),
        beamwidth_3db_deg=float(g["beamwidth_3db_deg"]),
        spacing_m=float(g["spacing_m"]),
        fc_hz=float(g["fc_hz"]),
        pattern_floor_db=float(g["pattern_floor_db"]),
    )


def two_ray_taps(rel_db: float | None, rng: np.random.Generator) -> list[MultipathTap]:
    if rel_db is None:
        return []
    amp = 10.0 ** (rel_db / 20.0)
    phase = float(rng.uniform(0.0, 2 * np.pi))
    return [MultipathTap(delay_s=TWO_RAY_DELAY_S,
                         gain=complex(amp * np.exp(1j * phase)),
                         theta_deg=TWO_RAY_REFL_DEG)]


def rayleigh_taps(rel_db: float | None, rng: np.random.Generator) -> list[MultipathTap]:
    if rel_db is None:
        return []
    total_amp = 10.0 ** (rel_db / 20.0)
    g = (rng.standard_normal(RAYLEIGH_N_TAPS)
         + 1j * rng.standard_normal(RAYLEIGH_N_TAPS)) / np.sqrt(2.0)
    g = g / np.sqrt(np.sum(np.abs(g) ** 2)) * total_amp     # normalise total power
    delays = np.linspace(RAYLEIGH_MAX_DELAY_S / RAYLEIGH_N_TAPS,
                         RAYLEIGH_MAX_DELAY_S, RAYLEIGH_N_TAPS)
    angles = rng.uniform(-60.0, 60.0, RAYLEIGH_N_TAPS)
    return [MultipathTap(delay_s=float(delays[i]), gain=complex(g[i]),
                         theta_deg=float(angles[i])) for i in range(RAYLEIGH_N_TAPS)]


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    fs = float(cfg["fs_hz"])
    geom = build_geometry(cfg)
    targets = [float(t) for t in cfg["theta_targets_deg"]]
    snr_db = float(cfg["simulator"]["snr_db"])
    fp_threshold = float(cfg["fingerprint"]["threshold"])
    fp_aggregate = str(cfg["fingerprint"]["aggregate"])
    bd = cfg["burst_detector"]
    avg_cfg = cfg["averager"]

    ff = FingerprintFilter.load(_resolve(cfg["fingerprint"]["model_path"]))
    ff.set_threshold(fp_threshold)

    # ---- Phase 1: detect + fingerprint once, cache kept bursts
    print("=== multipath sweep: caching kept bursts ===", flush=True)
    kept: list[tuple[np.ndarray, float]] = []   # (iq128, fc_offset_hz)
    t0 = time.time()
    for src in cfg["sources"]:
        iq_full = _load_iq_complex64(_resolve(src["path"]), float(src["duration_s"]), fs)
        bursts = detect_bursts(
            iq_full, fs, smoothing_us=float(bd["smoothing_us"]),
            threshold_db=float(bd["threshold_db"]), hysteresis_db=float(bd["hysteresis_db"]),
            min_duration_us=float(bd["min_duration_us"]),
            noise_quantile=float(bd["noise_quantile"]), stft_nfft=int(bd["stft_nfft"]),
        )
        for b in bursts:
            iq_burst = iq_full[b.t_start:b.t_end].astype(np.complex64)
            if ff.predict_burst_proba(iq_burst, aggregate=fp_aggregate) >= fp_threshold:
                kept.append((iq_burst.astype(np.complex128), float(b.fc_offset_hz)))
        print(f"  {src['name']}: {len(bursts)} bursts → {len(kept)} kept cumulatively",
              flush=True)
        del iq_full
    print(f"  cached {len(kept)} kept bursts in {time.time()-t0:.1f}s", flush=True)

    de = DifferentialEstimator()

    def sweep(make_taps) -> dict:
        out: dict[str, list] = {f"{th:+.0f}": [] for th in targets}
        for rel_db in REL_POWER_DB:
            taps = make_taps(rel_db, np.random.default_rng(TAP_SEED))
            for th in targets:
                av = Averager(window=int(avg_cfg["window"]), hac_lag=int(avg_cfg["hac_lag"]),
                              bucket_by_hop=bool(avg_cfg["bucket_by_hop"]),
                              bucket_step_hz=float(avg_cfg["bucket_step_hz"]))
                for k, (iq128, fc) in enumerate(kept):
                    sim = simulate_dual_channel(
                        iq128, theta_target_deg=th, fs=fs, geometry=geom,
                        snr_db=snr_db, multipath_taps=taps,
                        rng=np.random.default_rng(k + 1))
                    av.push(de.estimate(sim.iq_a, sim.iq_b), fc_offset_hz=fc)
                med = av.summary().median_db
                theta_hat = theta_from_delta_db(med, geom)
                out[f"{th:+.0f}"].append(round(float(theta_hat - th), 3))
        return out

    print("  sweeping 2-ray ...", flush=True)
    two_ray = sweep(two_ray_taps)
    print("  sweeping Rayleigh ...", flush=True)
    rayleigh = sweep(rayleigh_taps)

    x_db = [(-99.0 if r is None else r) for r in REL_POWER_DB]  # baseline -> -99 sentinel
    report = {
        "n_kept_bursts": len(kept),
        "rel_power_db_sweep": [("baseline" if r is None else r) for r in REL_POWER_DB],
        "two_ray": {"refl_deg": TWO_RAY_REFL_DEG, "delay_s": TWO_RAY_DELAY_S,
                    "err_deg_by_target": two_ray},
        "rayleigh": {"n_taps": RAYLEIGH_N_TAPS, "max_delay_s": RAYLEIGH_MAX_DELAY_S,
                     "err_deg_by_target": rayleigh},
        "snr_db": snr_db,
        "note": "Inverse uses the true geometry; only the multipath channel perturbs Δ.",
    }
    RESULTS.write_text(json.dumps(report, indent=2))
    print(f"saved → {RESULTS}", flush=True)
    for th in targets:
        k = f"{th:+.0f}"
        print(f"  θ={k}°  2-ray |err|@0dB={abs(two_ray[k][-1]):5.2f}°  "
              f"Rayleigh |err|@0dB={abs(rayleigh[k][-1]):5.2f}°", flush=True)

    # ---------------------------------------------------------------- figure
    palette = {"-45": "tab:red", "+0": "tab:green", "+30": "tab:orange",
               "+60": "tab:purple"}
    xs = x_db[1:]   # drop baseline sentinel from the x-axis; show as separate marker
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, data, title in ((ax_a, two_ray, "2-ray specular reflection"),
                            (ax_b, rayleigh, "Rayleigh diffuse scattering")):
        for th in targets:
            k = f"{th:+.0f}"
            ax.plot(xs, np.abs(data[k][1:]), marker="o", ms=3, lw=1.5,
                    color=palette.get(k, "k"), label=f"θ_target = {th:+.0f}°")
            ax.scatter([-18.0], [abs(data[k][0])], color=palette.get(k, "k"),
                       marker="s", s=28, zorder=5)
        ax.set_xlabel("relative reflected/scattered power (dB)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xticks([-18, -15, -12, -9, -6, -3, 0])
        ax.set_xticklabels(["clean", "-15", "-12", "-9", "-6", "-3", "0"])
    ax_a.set_ylabel("|θ̂ error| (deg)")
    ax_a.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Multipath robustness of the DOA proxy (SNR = {snr_db:.0f} dB, "
                 f"{len(kept)} kept bursts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)
    print(f"saved → {FIGURE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
