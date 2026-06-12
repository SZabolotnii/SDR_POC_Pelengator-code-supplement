#!/usr/bin/env python3
"""Conditioning + angular-uncertainty analysis of the amplitude-comparison DOA map.

Pure-geometry companion to the A5 proxy. It tabulates the differential beam
response Δ_dB(θ) and its sensitivity dΔ/dθ for the A5 antenna geometry, derives
the first-order angular uncertainty

    σ_θ(θ) ≈ SE_Δ / |dΔ/dθ| ,

and marks the well-conditioned vs saturated (beam-floored) sectors. This makes
explicit, independent of any forward/inverse round trip, *why* the A5 demo
angles −45/+30/+60° sit in a degenerate regime where the inverse is
ill-conditioned, while only θ≈0° is well conditioned.

Outputs:
  experiments/poc_pelengator/results/conditioning.json
  manuscript/hait_poc_pelengator/figures/conditioning.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sdr_kunchenko.doa.null_detector import predicted_delta_db_grid  # noqa: E402
from sdr_kunchenko.rf.dual_channel_simulator import (  # noqa: E402
    AntennaGeometry,
    cosine_beam_pattern,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG = HERE / "config.yaml"
RESULTS = HERE / "results" / "conditioning.json"
FIGURE = REPO_ROOT / "manuscript" / "hait_poc_pelengator" / "figures" / "conditioning.png"

# Conditioning threshold: below this slope a 1 dB measurement error already maps
# to > ~3° of angular error, which we treat as the edge of the usable sector.
SLOPE_THR_DB_PER_DEG = 0.30
# Representative differential-amplitude standard errors:
#   0.0009 dB — the A5 HAC SE on a clean AWGN proxy (Table A5);
#   0.5 / 1.0 dB — realistic field/calibration-grade measurement error.
SE_LEVELS_DB = [0.0009, 0.5, 1.0]


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


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    geom = build_geometry(cfg)
    targets = [float(t) for t in cfg["theta_targets_deg"]]

    grid = np.arange(-89.0, 89.0001, 0.02)
    _, delta = predicted_delta_db_grid(geom, theta_grid_deg=grid)
    slope = np.gradient(delta, grid)  # dΔ/dθ, dB per degree
    abs_slope = np.abs(slope)

    g_a = cosine_beam_pattern(grid, geom.boresight_a_deg, geom.beamwidth_3db_deg,
                              floor_db=geom.pattern_floor_db)
    g_b = cosine_beam_pattern(grid, geom.boresight_b_deg, geom.beamwidth_3db_deg,
                              floor_db=geom.pattern_floor_db)
    floor_lin = 10.0 ** (geom.pattern_floor_db / 10.0)
    a_floored = g_a <= floor_lin * 1.05
    b_floored = g_b <= floor_lin * 1.05
    saturated = a_floored | b_floored          # at least one channel railed
    well_cond = (abs_slope >= SLOPE_THR_DB_PER_DEG) & ~saturated

    # Contiguous well-conditioned interval that contains θ = 0.
    zero_idx = int(np.argmin(np.abs(grid)))
    lo = zero_idx
    while lo > 0 and well_cond[lo - 1]:
        lo -= 1
    hi = zero_idx
    while hi < len(grid) - 1 and well_cond[hi + 1]:
        hi += 1
    well_lo, well_hi = float(grid[lo]), float(grid[hi])

    def slope_at(theta: float) -> float:
        return float(abs_slope[int(np.argmin(np.abs(grid - theta)))])

    def delta_at(theta: float) -> float:
        return float(delta[int(np.argmin(np.abs(grid - theta)))])

    # Per-target conditioning table.
    table = []
    for th in targets + [-30.0, 30.0]:
        s = slope_at(th)
        row = {
            "theta_deg": th,
            "delta_db": round(delta_at(th), 4),
            "abs_slope_db_per_deg": round(s, 4),
            "saturated": bool(saturated[int(np.argmin(np.abs(grid - th)))]),
            "well_conditioned": bool(well_cond[int(np.argmin(np.abs(grid - th)))]),
            "sigma_theta_deg": {
                f"{se:g}dB": (round(se / s, 3) if s > 1e-9 else None)
                for se in SE_LEVELS_DB
            },
        }
        table.append(row)

    report = {
        "geometry": {
            "boresight_a_deg": geom.boresight_a_deg,
            "boresight_b_deg": geom.boresight_b_deg,
            "beamwidth_3db_deg": geom.beamwidth_3db_deg,
            "pattern_floor_db": geom.pattern_floor_db,
        },
        "slope_threshold_db_per_deg": SLOPE_THR_DB_PER_DEG,
        "se_levels_db": SE_LEVELS_DB,
        "well_conditioned_interval_deg": [well_lo, well_hi],
        "max_abs_slope_db_per_deg": round(float(abs_slope.max()), 4),
        "theta_at_max_slope_deg": round(float(grid[int(np.argmax(abs_slope))]), 2),
        "per_theta": table,
    }
    RESULTS.write_text(json.dumps(report, indent=2))
    print(f"saved → {RESULTS}")
    print(f"  well-conditioned interval: [{well_lo:.1f}, {well_hi:.1f}]°")
    for r in table:
        print(f"  θ={r['theta_deg']:+5.1f}°  Δ={r['delta_db']:+7.2f} dB  "
              f"|dΔ/dθ|={r['abs_slope_db_per_deg']:.3f} dB/°  "
              f"σ_θ(1 dB)={r['sigma_theta_deg']['1dB']}°  "
              f"{'SATURATED' if r['saturated'] else 'ok'}")

    # ---------------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True)

    # shade saturated sectors (draw contiguous runs)
    def shade(ax):
        in_run = False
        start = grid[0]
        for i in range(len(grid)):
            if saturated[i] and not in_run:
                in_run, start = True, grid[i]
            elif not saturated[i] and in_run:
                ax.axvspan(start, grid[i], color="0.85", zorder=0,
                           label="_nolegend_")
                in_run = False
        if in_run:
            ax.axvspan(start, grid[-1], color="0.85", zorder=0)

    shade(ax1)
    ax1.plot(grid, delta, color="navy", lw=1.6)
    ax1.axvspan(well_lo, well_hi, color="tab:green", alpha=0.10,
                label=f"well-conditioned [{well_lo:.0f}, {well_hi:.0f}]°")
    palette = {-45.0: "tab:red", 0.0: "tab:green", 30.0: "tab:orange",
               60.0: "tab:purple"}
    for th in targets:
        d = delta_at(th)
        ax1.scatter([th], [d], color=palette.get(th, "k"), zorder=5, s=42)
        ax1.annotate(f"{th:+.0f}°", (th, d), textcoords="offset points",
                     xytext=(4, 6), fontsize=8, color=palette.get(th, "k"))
    ax1.set_ylabel("Δ_dB(θ)")
    ax1.set_title("Differential beam response and its conditioning")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=7)

    shade(ax2)
    ax2.semilogy(grid, np.maximum(abs_slope, 1e-4), color="darkred", lw=1.4)
    ax2.axhline(SLOPE_THR_DB_PER_DEG, color="tab:green", ls="--", lw=1.0,
                label=f"slope threshold {SLOPE_THR_DB_PER_DEG} dB/°")
    for th in targets:
        s = slope_at(th)
        ax2.scatter([th], [max(s, 1e-4)], color=palette.get(th, "k"),
                    zorder=5, s=42)
        sig = (1.0 / s) if s > 1e-9 else np.inf
        txt = "σ_θ→∞" if not np.isfinite(sig) or sig > 90 else f"σ_θ={sig:.1f}°"
        ax2.annotate(txt, (th, max(s, 1e-4)), textcoords="offset points",
                     xytext=(4, 6), fontsize=7, color=palette.get(th, "k"))
    ax2.set_xlabel("θ (deg)")
    ax2.set_ylabel("|dΔ/dθ|  (dB/°)")
    ax2.set_xlim(-90, 90)
    ax2.set_xticks(np.arange(-90, 91, 30))
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.text(0.99, 0.04,
             "σ_θ annotations assume a 1 dB differential-amplitude error",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=6.5,
             style="italic", color="0.3")

    fig.tight_layout()
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)
    print(f"saved → {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
