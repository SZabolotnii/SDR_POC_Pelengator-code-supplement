#!/usr/bin/env python3
"""Antenna-pattern model-mismatch sensitivity of the differential-amplitude DOA.

The A5 proxy inverts the angle with the *same* cos² beam model the simulator
used, so its round-trip error is ~0 by construction. That is a pipeline-
consistency check, not an accuracy result. This script removes the tautology:
the forward Δ_dB is generated with the true geometry, but the inverse map uses a
*perturbed* (mis-calibrated) geometry. The resulting θ̂ error is therefore real
and quantifies how sensitive amplitude-comparison DOA is to imperfect knowledge
of the antenna front-end.

Three independent calibration-error axes:
  * channel gain imbalance  g (dB) added to the measured Δ;
  * boresight mispointing   δφ (deg): the whole array is rotated;
  * beamwidth error         δBW (%): the assumed 3-dB beamwidth is off.

Real-data grounding: the per-angle measured Δ_dB from `a5_proxy_report.json`
match the analytic forward model to < 1e-2 dB (Table A5), so the sensitivity
computed here transfers directly to the real Tampere bursts; the four A5
operating points are overlaid on the figure.

Outputs:
  experiments/poc_pelengator/results/mismatch_sensitivity.json
  manuscript/hait_poc_pelengator/figures/mismatch_sensitivity.png
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sdr_kunchenko.doa.null_detector import theta_from_delta_db  # noqa: E402
from sdr_kunchenko.rf.dual_channel_simulator import (  # noqa: E402
    AntennaGeometry,
    cosine_beam_pattern,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG = HERE / "config.yaml"
A5_REPORT = HERE / "results" / "a5_proxy_report.json"
RESULTS = HERE / "results" / "mismatch_sensitivity.json"
FIGURE = REPO_ROOT / "manuscript" / "hait_poc_pelengator" / "figures" / "mismatch_sensitivity.png"

# Fixed "typical calibration error" used for the error-vs-angle panel.
GAIN_FIX_DB = 1.0
BORESIGHT_FIX_DEG = 2.0
BEAMWIDTH_FIX_FRAC = 0.10

# Sweep ranges for the error-vs-magnitude panel (symmetric: a positive imbalance
# pins the saturated +30° plateau at its rail, so only a symmetric sweep with a
# *signed* error reveals the degeneracy honestly).
GAIN_SWEEP_DB = np.linspace(-3.0, 3.0, 31)


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


def delta_true(theta_deg: float, geom: AntennaGeometry) -> float:
    """Forward differential beam response (dB) at one angle."""
    g_a = cosine_beam_pattern(theta_deg, geom.boresight_a_deg,
                              geom.beamwidth_3db_deg, floor_db=geom.pattern_floor_db)
    g_b = cosine_beam_pattern(theta_deg, geom.boresight_b_deg,
                              geom.beamwidth_3db_deg, floor_db=geom.pattern_floor_db)
    return float(10.0 * np.log10(float(g_a) / float(g_b)))


def err_gain(theta, geom, g_db):
    return theta_from_delta_db(delta_true(theta, geom) + g_db, geom) - theta


def err_boresight(theta, geom, dphi):
    assumed = dataclasses.replace(
        geom, boresight_a_deg=geom.boresight_a_deg + dphi,
        boresight_b_deg=geom.boresight_b_deg + dphi)
    return theta_from_delta_db(delta_true(theta, geom), assumed) - theta


def err_beamwidth(theta, geom, frac):
    assumed = dataclasses.replace(
        geom, beamwidth_3db_deg=geom.beamwidth_3db_deg * (1.0 + frac))
    return theta_from_delta_db(delta_true(theta, geom), assumed) - theta


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    geom = build_geometry(cfg)
    targets = [float(t) for t in cfg["theta_targets_deg"]]

    # --- real-data grounding: analytic vs A5 measured Δ
    grounding = []
    if A5_REPORT.exists():
        a5 = json.loads(A5_REPORT.read_text())
        for row in a5.get("per_theta", []):
            th = float(row["theta_target_deg"])
            meas = float(row["delta_db_median_final"])
            ana = delta_true(th, geom)
            grounding.append({
                "theta_deg": th, "delta_measured_db": round(meas, 4),
                "delta_analytic_db": round(ana, 4),
                "abs_diff_db": round(abs(meas - ana), 5),
            })

    # --- error-vs-angle at fixed typical mismatch
    th_grid = np.arange(-65.0, 65.0001, 0.5)
    e_gain = np.array([err_gain(t, geom, GAIN_FIX_DB) for t in th_grid])
    e_bore = np.array([err_boresight(t, geom, BORESIGHT_FIX_DEG) for t in th_grid])
    e_bw = np.array([err_beamwidth(t, geom, BEAMWIDTH_FIX_FRAC) for t in th_grid])

    well = np.abs(th_grid) <= 30.0
    summary = {
        "gain_1dB": {
            "mean_abs_err_well_cond_deg": round(float(np.mean(np.abs(e_gain[well]))), 3),
            "mean_abs_err_saturated_deg": round(float(np.mean(np.abs(e_gain[~well]))), 3),
            "max_abs_err_deg": round(float(np.max(np.abs(e_gain))), 3),
        },
        "boresight_2deg": {
            "mean_abs_err_well_cond_deg": round(float(np.mean(np.abs(e_bore[well]))), 3),
            "mean_abs_err_saturated_deg": round(float(np.mean(np.abs(e_bore[~well]))), 3),
            "max_abs_err_deg": round(float(np.max(np.abs(e_bore))), 3),
        },
        "beamwidth_10pct": {
            "mean_abs_err_well_cond_deg": round(float(np.mean(np.abs(e_bw[well]))), 3),
            "mean_abs_err_saturated_deg": round(float(np.mean(np.abs(e_bw[~well]))), 3),
            "max_abs_err_deg": round(float(np.max(np.abs(e_bw))), 3),
        },
    }

    # --- error-vs-magnitude (gain) at the four demo angles (signed error)
    gain_curves = {}
    for th in targets:
        gain_curves[f"{th:+.0f}"] = [
            round(float(err_gain(th, geom, g)), 3) for g in GAIN_SWEEP_DB
        ]

    report = {
        "fixed_mismatch": {
            "gain_db": GAIN_FIX_DB, "boresight_deg": BORESIGHT_FIX_DEG,
            "beamwidth_frac": BEAMWIDTH_FIX_FRAC,
        },
        "gain_sweep_db": [round(float(g), 3) for g in GAIN_SWEEP_DB],
        "gain_signed_err_deg_by_target": gain_curves,
        "error_vs_angle_summary": summary,
        "real_data_grounding": grounding,
        "note": (
            "Forward Δ uses the true geometry; the inverse uses a perturbed "
            "geometry, so these θ̂ errors are real (not the A5 tautological 0°)."
        ),
    }
    RESULTS.write_text(json.dumps(report, indent=2))
    print(f"saved → {RESULTS}")
    for name, s in summary.items():
        print(f"  {name:16s}  well-cond mean |err|={s['mean_abs_err_well_cond_deg']:5.2f}°  "
              f"saturated mean |err|={s['mean_abs_err_saturated_deg']:6.2f}°  "
              f"max={s['max_abs_err_deg']:6.2f}°")
    if grounding:
        max_gap = max(r["abs_diff_db"] for r in grounding)
        print(f"  real-data grounding: max |Δ_measured − Δ_analytic| = {max_gap:.4f} dB")

    # ---------------------------------------------------------------- figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax_a.axvspan(-30, 30, color="tab:green", alpha=0.10, label="well-conditioned")
    ax_a.plot(th_grid, e_gain, color="tab:blue", lw=1.6,
             label=f"gain imbalance {GAIN_FIX_DB:.0f} dB")
    ax_a.plot(th_grid, e_bore, color="tab:orange", lw=1.6,
             label=f"boresight error {BORESIGHT_FIX_DEG:.0f}°")
    ax_a.plot(th_grid, e_bw, color="tab:purple", lw=1.6,
             label=f"beamwidth error {BEAMWIDTH_FIX_FRAC*100:.0f}%")
    ax_a.axhline(0, color="0.5", lw=0.8)
    for th in targets:
        ax_a.axvline(th, color="0.7", ls=":", lw=0.8)
    ax_a.set_xlabel("true angle θ (deg)")
    ax_a.set_ylabel("θ̂ error (deg)")
    ax_a.set_title("Angle error under a fixed calibration mismatch")
    ax_a.set_xlim(-65, 65)
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="upper center", fontsize=7.5)

    palette = {"-45": "tab:red", "+0": "tab:green", "+30": "tab:orange",
               "+60": "tab:purple"}
    for th in targets:
        key = f"{th:+.0f}"
        ax_b.plot(GAIN_SWEEP_DB, gain_curves[key], lw=1.7,
                 color=palette.get(key, "k"), label=f"θ_target = {th:+.0f}°")
    ax_b.axhline(0, color="0.5", lw=0.8)
    ax_b.axvline(0, color="0.5", lw=0.8)
    ax_b.set_xlabel("channel gain imbalance (dB)")
    ax_b.set_ylabel("θ̂ error (deg)")
    ax_b.set_title("Error vs gain imbalance at the A5 demo angles")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)
    print(f"saved → {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
