#!/usr/bin/env python3
"""A4 acceptance run — θ_target ∈ {−60, −30, 0, 30, 60} → θ̂ within ±5°.

Pipes A2 dual_channel_simulator through A4 DifferentialEstimator +
Averager + NullDetector + theta_from_delta_db, repeated for each angle
across many independent IQ realisations. Acceptance follows plan §A4.

Geometry note
-------------
The A2 default geometry (φ = ±15°, BW₃ = 30°) puts everything beyond
~±45° in the blind sector, which would make θ_target = ±60° physically
unreachable. Realistic 2.4 GHz patch antennas have BW₃ ≈ 60–80° per
element, so for the acceptance test we use a wide-beam configuration
(φ = ±30°, BW₃ = 60°) — same array spacing, same array-phase term.
This is documented as the canonical pelengator front-end in the A4
context; the narrow-beam default in A2 stays for unit tests.

Outputs
-------
  experiments/poc_pelengator/results/a4_acceptance_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from sdr_kunchenko.doa import (
    Averager,
    DifferentialEstimator,
    NullDetector,
    theta_from_delta_db,
)
from sdr_kunchenko.rf.dual_channel_simulator import (
    AntennaGeometry,
    simulate_dual_channel,
)

OUT_DIR = Path(__file__).resolve().parent / "results"

THETA_TARGETS_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)
N_PAIRS = 60                # bursts averaged per angle
WINDOW = 50                 # rolling window for the averager
HAC_LAG = 5
SNR_DB = 25.0
ABS_DB_THR = 0.5
CI_WIDTH_THR = 1.0
MIN_CONSECUTIVE = 3
ACCEPT_DEG = 5.0
SEED = 0


def _make_carrier(n: int = 4096, fs: float = 10e6, f0: float = 250e3) -> np.ndarray:
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * f0 * t).astype(np.complex128)


def run_one(
    iq: np.ndarray, fs: float, theta_target_deg: float,
    geom: AntennaGeometry, *, n_pairs: int, snr_db: float,
    seed_base: int,
) -> dict:
    de = DifferentialEstimator()
    av = Averager(window=WINDOW, hac_lag=HAC_LAG)
    nd = NullDetector(
        abs_db_thr=ABS_DB_THR, ci_width_thr=CI_WIDTH_THR,
        min_consecutive=MIN_CONSECUTIVE,
    )

    final_summary = None
    null_after = None

    for k in range(n_pairs):
        res = simulate_dual_channel(
            iq, theta_target_deg=theta_target_deg, fs=fs, geometry=geom,
            snr_db=snr_db, rng=np.random.default_rng(seed_base + k),
        )
        est = de.estimate(res.iq_a, res.iq_b)
        s = av.push(est)
        triggered = nd.update(s)
        if triggered and null_after is None:
            null_after = k + 1
        final_summary = s

    if final_summary is None:
        raise RuntimeError("No estimates collected")

    theta_hat = theta_from_delta_db(final_summary.median_db, geom)
    err_deg = theta_hat - theta_target_deg
    accepts = abs(err_deg) <= ACCEPT_DEG

    return {
        "theta_target_deg": theta_target_deg,
        "theta_hat_deg": theta_hat,
        "abs_err_deg": float(abs(err_deg)),
        "delta_db_median": final_summary.median_db,
        "delta_db_se": final_summary.se_db,
        "delta_db_ci95": list(final_summary.ci95_db),
        "ci_width_db": final_summary.ci_width_db,
        "n_pairs": n_pairs,
        "null_triggered_after": null_after,
        "accepts": accepts,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    geom = AntennaGeometry(
        boresight_a_deg=30.0, boresight_b_deg=-30.0, beamwidth_3db_deg=60.0,
    )
    fs = 10e6
    iq = _make_carrier(n=4096, fs=fs, f0=250e3)

    print("=== A4 acceptance run ===", flush=True)
    print(
        f"  geometry: φ_a={geom.boresight_a_deg}°, φ_b={geom.boresight_b_deg}°, "
        f"BW₃={geom.beamwidth_3db_deg}°", flush=True,
    )
    print(
        f"  N={N_PAIRS} pairs/angle, SNR={SNR_DB} dB, "
        f"window={WINDOW}, HAC lag={HAC_LAG}", flush=True,
    )
    print(
        f"  null thresholds: |Δ̄| < {ABS_DB_THR} dB, CI width < {CI_WIDTH_THR} dB, "
        f"k = {MIN_CONSECUTIVE} consecutive", flush=True,
    )
    print(
        f"  acceptance: |θ̂ − θ_target| ≤ {ACCEPT_DEG}° for each angle\n",
        flush=True,
    )

    rows: list[dict] = []
    t0 = time.time()
    for j, theta in enumerate(THETA_TARGETS_DEG):
        seed_base = SEED + 1000 * j
        row = run_one(
            iq, fs, theta, geom,
            n_pairs=N_PAIRS, snr_db=SNR_DB, seed_base=seed_base,
        )
        rows.append(row)
        flag = "✓" if row["accepts"] else "✗"
        print(
            f"  {flag} θ={theta:+5.1f} → Δ̄={row['delta_db_median']:+7.2f} dB "
            f"± {row['delta_db_se']:.3f}  θ̂={row['theta_hat_deg']:+7.2f}  "
            f"err={row['abs_err_deg']:5.2f}°  null@={row['null_triggered_after']}",
            flush=True,
        )
    elapsed = time.time() - t0

    n_accept = sum(int(r["accepts"]) for r in rows)
    verdict = "PASS" if n_accept == len(rows) else "FAIL"
    print(
        f"\n  ACCEPTANCE: {verdict}  ({n_accept}/{len(rows)} angles "
        f"within ±{ACCEPT_DEG}°)  [{elapsed:.1f}s]",
        flush=True,
    )

    report = {
        "config": {
            "theta_targets_deg": list(THETA_TARGETS_DEG),
            "n_pairs": N_PAIRS, "window": WINDOW, "hac_lag": HAC_LAG,
            "snr_db": SNR_DB, "seed": SEED,
            "abs_db_thr": ABS_DB_THR, "ci_width_thr": CI_WIDTH_THR,
            "min_consecutive": MIN_CONSECUTIVE, "accept_deg": ACCEPT_DEG,
        },
        "geometry": {
            "boresight_a_deg": geom.boresight_a_deg,
            "boresight_b_deg": geom.boresight_b_deg,
            "beamwidth_3db_deg": geom.beamwidth_3db_deg,
            "spacing_m": geom.spacing_m,
            "fc_hz": geom.fc_hz,
            "pattern_floor_db": geom.pattern_floor_db,
        },
        "rows": rows,
        "n_accept": n_accept,
        "n_total": len(rows),
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 2),
    }
    out = OUT_DIR / "a4_acceptance_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  saved → {out}", flush=True)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
