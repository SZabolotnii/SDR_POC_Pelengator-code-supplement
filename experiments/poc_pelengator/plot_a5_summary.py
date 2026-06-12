#!/usr/bin/env python3
"""Consolidated A5 proxy figure for the manuscript (replaces four flat traces).

Reads the A5 per-burst log and renders a single two-panel figure covering all
four target angles:
  * top    — per-burst Δ_dB(t) + rolling median for each angle (the observable);
  * bottom — θ̂(t) convergence for each angle with its ±accept_deg band (degrees)
             and the θ=0 null-trigger marker.

This is an end-to-end pipeline-consistency view (the inverse uses the same beam
model as the simulator), not a field-accuracy plot — see the conditioning and
mismatch figures for the real-error analysis.

Output:
  manuscript/hait_poc_pelengator/figures/a5_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sdr_kunchenko.doa.null_detector import theta_from_delta_db  # noqa: E402
from sdr_kunchenko.rf.dual_channel_simulator import AntennaGeometry  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG = HERE / "config.yaml"
LOG = HERE / "results" / "a5_proxy_log.jsonl"
REPORT = HERE / "results" / "a5_proxy_report.json"
FIGURE = REPO_ROOT / "manuscript" / "hait_poc_pelengator" / "figures" / "a5_summary.png"

PALETTE = {-45.0: "tab:red", 0.0: "tab:green", 30.0: "tab:orange", 60.0: "tab:purple"}


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
    accept_deg = float(cfg["acceptance"]["accept_deg"])

    report = json.loads(REPORT.read_text())
    null_t = {float(r["theta_target_deg"]): r["null_triggered_at_t_s"]
              for r in report["per_theta"]}
    n_kept = report["n_kept_total"]
    n_total = report["n_total_bursts"]

    # per-angle time series, plus source bookkeeping for the Inspire-2 leak note
    series = {th: {"t": [], "raw": [], "med": [], "theta": []} for th in targets}
    src_change_t = None
    n_inspire_kept = 0
    prev_src = None
    for line in LOG.read_text().splitlines():
        row = json.loads(line)
        if not row.get("kept"):
            continue
        t = float(row["t_global_s"])
        src = row.get("src", "")
        if prev_src is not None and src != prev_src and src_change_t is None:
            src_change_t = t
        prev_src = src
        if src != "DJI_Mavic_Pro":
            n_inspire_kept += 1
        for entry in row["thetas"]:
            th = float(entry["theta_target_deg"])
            if th not in series:
                continue
            series[th]["t"].append(t)
            series[th]["raw"].append(float(entry["delta_db"]))
            series[th]["med"].append(float(entry["median_db"]))
            series[th]["theta"].append(theta_from_delta_db(float(entry["median_db"]), geom))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    # ---- top: Δ_dB(t)
    for th in targets:
        s = series[th]
        c = PALETTE.get(th, "k")
        ax1.scatter(s["t"], s["raw"], s=8, alpha=0.35, color=c, edgecolors="none")
        ax1.plot(s["t"], s["med"], lw=1.5, color=c, label=f"θ={th:+.0f}°")
    # Shade the post-boundary region where the Inspire-2 false positives enter:
    # the estimate stays flat across it, which is the actual robustness evidence.
    if src_change_t is not None:
        x_hi = ax1.get_xlim()[1]
        for ax in (ax1, ax2):
            ax.axvspan(src_change_t, x_hi, color="0.85", alpha=0.35, zorder=0)
        ax1.axvline(src_change_t, color="0.5", ls=":", lw=1.0)
        ax1.text(src_change_t, ax1.get_ylim()[1],
                 f"  source boundary → {n_inspire_kept} Inspire-2 FPs",
                 ha="left", va="top", fontsize=6.5, color="0.4")
    # The two saturated runs (+30°, +60°) both pin near the +60 dB ceiling: one
    # channel sits at the pattern floor, so |dΔ/dθ|→0 and the inverse is
    # ill-conditioned there — the real-error driver analysed in the conditioning
    # figure, not visible from the flat traces alone.
    ax1.annotate("+30° & +60° pin near the +60 dB ceiling\n"
                 "(saturated, ill-conditioned — see conditioning fig.)",
                 xy=(0.02, 0.93), xycoords="axes fraction",
                 ha="left", va="top", fontsize=6.0, color="0.30",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5, alpha=0.85))
    ax1.set_ylabel("Δ_dB(t)")
    ax1.set_title(f"End-to-end proxy — consistency over 4 target angles "
                  f"({n_kept}/{n_total} bursts kept)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="center left", fontsize=7.5, ncol=2)

    # ---- bottom: theta_hat(t) with degree-correct accept bands
    for th in targets:
        s = series[th]
        c = PALETTE.get(th, "k")
        ax2.axhspan(th - accept_deg, th + accept_deg, color=c, alpha=0.10)
        ax2.plot(s["t"], s["theta"], lw=1.7, color=c, label=f"θ̂→{th:+.0f}°")
    for _th, tn in null_t.items():
        if tn is not None:
            ax2.axvline(tn, color="tab:green", ls="--", lw=1.0)
            ax2.text(tn, -85, f" null @ θ=0, t={tn*1e3:.0f} ms",
                     ha="left", va="bottom", fontsize=6.5, color="tab:green")
    if src_change_t is not None:
        ax2.axvline(src_change_t, color="0.5", ls=":", lw=1.0)
    ax2.set_xlabel("burst time (s) — the four target-angle runs share one axis")
    ax2.set_ylabel("θ̂ (deg)")
    ax2.set_ylim(-90, 90)
    ax2.set_yticks(np.arange(-90, 91, 30))
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center left", fontsize=7.5, ncol=2)

    fig.text(0.5, 0.005,
             f"Bands are the ±{accept_deg:.0f}° acceptance windows (degrees). "
             f"{n_inspire_kept}/{n_kept} kept bursts are Inspire-2 false positives "
             f"(after the source boundary).",
             ha="center", va="bottom", fontsize=7.0, style="italic", color="0.3")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)
    print(f"saved → {FIGURE}  (Inspire-2 kept: {n_inspire_kept}/{n_kept})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
