#!/usr/bin/env python3
"""Phase 7 / A5 — end-to-end pelengator proxy demo on real Tampere I/Q.

Pipeline (all real-bursts, only the dual-channel geometry is synthetic):
    real concat IQ → A0 packet_detector → A3 fingerprint_filter (keep
    Mavic Pro) → A2 dual_channel_simulator(θ_target) → A4 DifferentialEstimator
    → A4 Averager → A4 NullDetector + theta_from_delta_db.

For each θ_target ∈ config.theta_targets_deg:
    * Δ̄(t) trace (matplotlib PNG)
    * one-row entry in the per-θ summary
The per-burst log goes to a single JSONL covering all θ.

Acceptance (plan §A5): ≥3 of 4 θ_target dgive |θ̂ − θ_target| ≤ 5°.

Outputs (paths overridable in config):
    experiments/poc_pelengator/results/a5_proxy_log.jsonl
    experiments/poc_pelengator/results/a5_proxy_report.json
    experiments/poc_pelengator/figures/a5_proxy_theta_{θ}.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sdr_kunchenko.doa import (
    Averager,
    DifferentialEstimator,
    NullDetector,
    theta_from_delta_db,
)
from sdr_kunchenko.rf.dual_channel_simulator import (
    AntennaGeometry,
    MultipathTap,
    simulate_dual_channel,
)
from sdr_kunchenko.rf.fingerprint import FingerprintFilter
from sdr_kunchenko.rf.packet_detector import detect_bursts

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- IO


def _load_iq_complex64(path: Path, duration_s: float, fs: float) -> np.ndarray:
    """Read int16 LE I/Q from `path`, return complex64 of length round(duration_s · fs)."""
    n_complex = int(round(duration_s * fs))
    data = np.fromfile(path, dtype="<i2", count=2 * n_complex)
    iq = data.astype(np.float32).reshape(-1, 2)
    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


# ---------------------------------------------------------------- pipeline


def _build_geometry(d: dict) -> AntennaGeometry:
    return AntennaGeometry(
        boresight_a_deg=float(d["boresight_a_deg"]),
        boresight_b_deg=float(d["boresight_b_deg"]),
        beamwidth_3db_deg=float(d["beamwidth_3db_deg"]),
        spacing_m=float(d["spacing_m"]),
        fc_hz=float(d["fc_hz"]),
        pattern_floor_db=float(d["pattern_floor_db"]),
    )


def _build_multipath_taps(taps_cfg: list[dict]) -> list[MultipathTap]:
    out: list[MultipathTap] = []
    for tap in taps_cfg or []:
        out.append(MultipathTap(
            delay_s=float(tap["delay_s"]),
            gain=complex(float(tap["gain_re"]), float(tap["gain_im"])),
            theta_deg=float(tap["theta_deg"]),
        ))
    return out


def _per_theta_state(theta_deg: float, cfg_avg: dict, cfg_null: dict) -> dict:
    return {
        "theta_target_deg": theta_deg,
        "averager": Averager(
            window=int(cfg_avg["window"]), hac_lag=int(cfg_avg["hac_lag"]),
            bucket_by_hop=bool(cfg_avg["bucket_by_hop"]),
            bucket_step_hz=float(cfg_avg["bucket_step_hz"]),
        ),
        "null_detector": NullDetector(
            abs_db_thr=float(cfg_null["abs_db_thr"]),
            ci_width_thr=float(cfg_null["ci_width_thr"]),
            min_consecutive=int(cfg_null["min_consecutive"]),
        ),
        "trace_t": [],            # global timestamps (kept bursts only)
        "trace_raw": [],          # per-burst Δ_dB before averaging
        "trace_median": [],       # rolling median Δ̄ at each kept-burst step
        "trace_se": [],           # HAC SE on the median series
        "trace_theta_hat": [],    # θ̂(t) — running inverse of trace_median
        "trace_src": [],          # source name per kept burst
        "source_boundaries_s": [],   # cumulative time at the end of each source
        "null_first_t": None,
        "n_kept": 0,
    }


def run(config_path: Path) -> int:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    fs = float(cfg["fs_hz"])
    theta_targets = [float(t) for t in cfg["theta_targets_deg"]]
    geom = _build_geometry(cfg["geometry"])
    taps = _build_multipath_taps(cfg["simulator"].get("multipath_taps") or [])
    snr_db = float(cfg["simulator"]["snr_db"])
    fp_threshold = float(cfg["fingerprint"]["threshold"])
    fp_aggregate = str(cfg["fingerprint"]["aggregate"])

    results_dir = _resolve(cfg["output"]["results_dir"])
    figures_dir = _resolve(cfg["output"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / cfg["output"]["log_jsonl"]
    summary_path = results_dir / cfg["output"]["summary_json"]
    fig_prefix = cfg["output"]["figure_prefix"]

    print("=== A5 pelengator proxy ===", flush=True)
    print(f"  fs={fs:.1e}, θ_targets={theta_targets}, snr_db={snr_db}", flush=True)
    print(
        f"  geom: φ={geom.boresight_a_deg}/{geom.boresight_b_deg}°, "
        f"BW₃={geom.beamwidth_3db_deg}°, spacing={geom.spacing_m*100:.1f} cm",
        flush=True,
    )

    # ----- fingerprint
    fp_path = _resolve(cfg["fingerprint"]["model_path"])
    print(f"  loading fingerprint: {fp_path}", flush=True)
    ff = FingerprintFilter.load(fp_path)
    ff.set_threshold(fp_threshold)

    # ----- per-θ state
    states = [_per_theta_state(th, cfg["averager"], cfg["null_detector"])
              for th in theta_targets]

    de = DifferentialEstimator()
    bd_cfg = cfg["burst_detector"]

    # ----- single global JSONL log
    log_f = log_path.open("w")
    rows_written = 0

    n_total_bursts = 0
    n_kept_total = 0
    global_t0 = 0.0     # cumulative time origin across sources

    t_start_wall = time.time()
    for src in cfg["sources"]:
        src_name = str(src["name"])
        src_path = _resolve(src["path"])
        src_dur = float(src["duration_s"])
        print(f"\n  loading {src_name}: {src_path} ({src_dur:.2f}s)", flush=True)
        iq_full = _load_iq_complex64(src_path, src_dur, fs)
        n_samples = iq_full.size

        bursts = detect_bursts(
            iq_full, fs,
            smoothing_us=float(bd_cfg["smoothing_us"]),
            threshold_db=float(bd_cfg["threshold_db"]),
            hysteresis_db=float(bd_cfg["hysteresis_db"]),
            min_duration_us=float(bd_cfg["min_duration_us"]),
            noise_quantile=float(bd_cfg["noise_quantile"]),
            stft_nfft=int(bd_cfg["stft_nfft"]),
        )
        print(f"    bursts detected: {len(bursts)}", flush=True)

        for b in bursts:
            n_total_bursts += 1
            iq_burst = iq_full[b.t_start : b.t_end].astype(np.complex64)
            ts = global_t0 + b.t_start / fs

            p_mavic = ff.predict_burst_proba(
                iq_burst, aggregate=fp_aggregate,
            )
            kept = bool(p_mavic >= fp_threshold)
            if not kept:
                log_f.write(json.dumps({
                    "src": src_name, "t_global_s": float(ts),
                    "fc_offset_hz": float(b.fc_offset_hz),
                    "p_mavic": float(p_mavic), "kept": False,
                }) + "\n")
                rows_written += 1
                continue

            n_kept_total += 1
            # We need complex128 for the simulator (float64 inside).
            iq128 = iq_burst.astype(np.complex128)

            row_thetas: list[dict[str, Any]] = []
            for st in states:
                theta_target = st["theta_target_deg"]
                sim = simulate_dual_channel(
                    iq128, theta_target_deg=theta_target, fs=fs, geometry=geom,
                    snr_db=snr_db, multipath_taps=taps,
                    rng=np.random.default_rng(n_kept_total),
                )
                est = de.estimate(sim.iq_a, sim.iq_b)
                summary = st["averager"].push(
                    est, fc_offset_hz=float(b.fc_offset_hz),
                )
                triggered = st["null_detector"].update(summary)
                if triggered and st["null_first_t"] is None:
                    st["null_first_t"] = float(ts)
                st["trace_t"].append(float(ts))
                st["trace_raw"].append(float(est.delta_db))
                st["trace_median"].append(float(summary.median_db))
                st["trace_se"].append(float(summary.se_db))
                st["trace_src"].append(src_name)
                # Cumulative θ̂ trajectory: invert the running median at every step.
                st["trace_theta_hat"].append(
                    float(theta_from_delta_db(summary.median_db, geom))
                )
                st["n_kept"] += 1
                row_thetas.append({
                    "theta_target_deg": theta_target,
                    "delta_db": float(est.delta_db),
                    "median_db": float(summary.median_db),
                    "se_db": float(summary.se_db),
                    "null_triggered": bool(triggered),
                })
            log_f.write(json.dumps({
                "src": src_name, "t_global_s": float(ts),
                "fc_offset_hz": float(b.fc_offset_hz),
                "p_mavic": float(p_mavic), "kept": True,
                "thetas": row_thetas,
            }) + "\n")
            rows_written += 1

        global_t0 += n_samples / fs    # advance to the next concatenated source
        for st in states:
            st["source_boundaries_s"].append(float(global_t0))
        del iq_full

    log_f.close()
    elapsed = time.time() - t_start_wall

    # ----- per-θ acceptance + plot
    accept_deg = float(cfg["acceptance"]["accept_deg"])
    min_pass = int(cfg["acceptance"]["min_pass_count"])

    per_theta_rows: list[dict] = []
    for st in states:
        theta_target = st["theta_target_deg"]
        if st["n_kept"] == 0:
            theta_hat = float("nan")
            err_deg = float("nan")
            accepts = False
            final_median = float("nan")
            final_se = float("nan")
        else:
            final_median = st["trace_median"][-1]
            final_se = st["trace_se"][-1]
            theta_hat = theta_from_delta_db(final_median, geom)
            err_deg = abs(theta_hat - theta_target)
            accepts = err_deg <= accept_deg
        per_theta_rows.append({
            "theta_target_deg": theta_target,
            "n_kept": st["n_kept"],
            "delta_db_median_final": final_median,
            "delta_db_se_final": final_se,
            "theta_hat_deg": theta_hat,
            "abs_err_deg": float(err_deg),
            "null_triggered_at_t_s": st["null_first_t"],
            "accepts": accepts,
        })
        _plot_trace(st, geom, accept_deg, figures_dir / f"{fig_prefix}_{int(theta_target):+d}.png")

    n_accept = sum(int(r["accepts"]) for r in per_theta_rows)
    verdict = "PASS" if n_accept >= min_pass else "FAIL"
    print("\n  per-θ result:", flush=True)
    for r in per_theta_rows:
        flag = "✓" if r["accepts"] else "✗"
        print(
            f"    {flag} θ={r['theta_target_deg']:+5.1f}°  n_kept={r['n_kept']:>4d}  "
            f"Δ̄={r['delta_db_median_final']:+7.2f} dB ± {r['delta_db_se_final']:.3f}  "
            f"θ̂={r['theta_hat_deg']:+7.2f}°  err={r['abs_err_deg']:5.2f}°  "
            f"null@t={r['null_triggered_at_t_s']}",
            flush=True,
        )
    print(
        f"\n  ACCEPTANCE: {verdict}  ({n_accept}/{len(per_theta_rows)} angles within "
        f"±{accept_deg}°; need ≥{min_pass}; n_total_bursts={n_total_bursts}, "
        f"n_kept={n_kept_total}; {elapsed:.1f}s wall)",
        flush=True,
    )

    summary_path.write_text(json.dumps({
        "config_path": str(config_path),
        "config": cfg,
        "n_total_bursts": n_total_bursts,
        "n_kept_total": n_kept_total,
        "log_jsonl_rows": rows_written,
        "per_theta": per_theta_rows,
        "n_accept": n_accept,
        "min_pass_count": min_pass,
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 2),
    }, indent=2))
    print(f"  saved → {summary_path}", flush=True)
    print(f"  log   → {log_path}", flush=True)

    return 0 if verdict == "PASS" else 1


def _plot_trace(state: dict, geom: AntennaGeometry, accept_deg: float, out_path: Path) -> None:
    """Two-row figure for one θ_target.

    Row 1 — per-burst Δ scatter + rolling median Δ̄ + 95 % CI band, with
            source boundaries and null trigger overlaid. Once AWGN is the
            only nuisance and the beam pattern is fixed, per-burst Δ_dB
            is nearly constant — the scatter colours surface the rare
            Inspire-2 false-positives that slip past the fingerprint.

    Row 2 — θ̂(t) convergence: at every kept burst we re-invert the
            current median Δ̄ through the cosine beam pattern. The first
            handful of bursts give a noisy θ̂ estimate, then the trace
            tightens onto θ_target ± accept_deg. This is the actual
            pelengator output a downstream OSD would consume.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sdr_kunchenko.doa.null_detector import predicted_delta_db_grid

    theta_target = state["theta_target_deg"]
    t = np.asarray(state["trace_t"])
    raw = np.asarray(state["trace_raw"])
    median = np.asarray(state["trace_median"])
    se = np.asarray(state["trace_se"])
    theta_hat = np.asarray(state.get("trace_theta_hat", []))
    src = state.get("trace_src", [])
    src_arr = np.asarray(src) if src else np.array([], dtype=str)

    grid, dgrid = predicted_delta_db_grid(geom)
    target_idx = int(np.argmin(np.abs(grid - theta_target)))
    delta_target = float(dgrid[target_idx])
    lo_idx = int(np.argmin(np.abs(grid - (theta_target - accept_deg))))
    hi_idx = int(np.argmin(np.abs(grid - (theta_target + accept_deg))))
    delta_lo = float(min(dgrid[lo_idx], dgrid[hi_idx]))
    delta_hi = float(max(dgrid[lo_idx], dgrid[hi_idx]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # ---------------------------- top: Δ_dB
    if t.size > 0:
        if src_arr.size:
            palette = {"DJI_Mavic_Pro": "tab:blue", "DJI_Inspire_2": "tab:orange"}
            for src_name in np.unique(src_arr):
                m = src_arr == src_name
                ax1.scatter(
                    t[m], raw[m], s=18, alpha=0.55,
                    color=palette.get(src_name, "tab:gray"),
                    edgecolors="none",
                    label=f"per-burst Δ ({src_name})",
                )
        else:
            ax1.scatter(t, raw, s=18, alpha=0.55, color="tab:gray",
                         edgecolors="none", label="per-burst Δ")

        ax1.fill_between(
            t, median - 1.96 * se, median + 1.96 * se,
            alpha=0.25, color="tab:blue", label="95 % CI of median",
        )
        ax1.plot(t, median, lw=1.6, color="navy",
                 label=f"rolling median Δ̄ (final {median[-1]:+.2f} dB)")

    ax1.axhline(delta_target, color="k", lw=1.0,
                label=f"Δ_target ({delta_target:+.2f} dB)")
    # This band lives on the dB axis: it is the Δ_dB interval that the ±accept_deg
    # angular tolerance maps to through the beam model (its width is therefore the
    # local |dΔ/dθ|·accept_deg, not a fixed dB number).
    ax1.axhspan(delta_lo, delta_hi, color="green", alpha=0.12,
                label=f"Δ band for ±{accept_deg:.0f}° (beam model)")
    for i, b_t in enumerate(state.get("source_boundaries_s", [])):
        ax1.axvline(b_t, color="purple", lw=0.7, ls=":", alpha=0.8,
                    label="source boundary" if i == 0 else None)
    if state["null_first_t"] is not None:
        ax1.axvline(
            state["null_first_t"], color="red", lw=1.0, ls="--",
            label=f"null trigger @ t={state['null_first_t']*1e3:.0f} ms",
        )
    ax1.set_ylabel("Δ_dB")
    ax1.set_title(
        f"Pelengator proxy — θ_target = {theta_target:+.0f}°  "
        f"({state['n_kept']} kept bursts)"
    )
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=7, ncol=2)
    # Adaptive Δ_dB y-range: keep both target and observed values comfortably
    # in view. The accept band can be very narrow at the saturated edges
    # (|θ| near beam null) — without an explicit margin matplotlib zooms in
    # so tight that the trace looks dead-flat regardless of variance.
    delta_margin = max(10.0, abs(delta_hi - delta_lo) * 4.0)
    ax1.set_ylim(delta_target - delta_margin, delta_target + delta_margin)

    # ---------------------------- bottom: θ̂(t)
    if theta_hat.size > 0:
        # Highlight the cold-start phase where the averager's window is not yet full.
        cold_start_n = min(state.get("n_kept", 0), 20)
        if cold_start_n > 0:
            ax2.plot(
                t[:cold_start_n], theta_hat[:cold_start_n], lw=1.8,
                color="tab:red", alpha=0.6,
                label=f"θ̂ cold-start (first {cold_start_n} bursts)",
            )
            ax2.plot(
                t[cold_start_n - 1 :], theta_hat[cold_start_n - 1 :], lw=1.8,
                color="tab:red",
                label=f"θ̂(t) (final {theta_hat[-1]:+.2f}°)",
            )
        else:
            ax2.plot(t, theta_hat, lw=1.8, color="tab:red", label="θ̂(t)")

    ax2.axhline(theta_target, color="k", lw=1.0,
                label=f"θ_target ({theta_target:+.0f}°)")
    ax2.axhspan(
        theta_target - accept_deg, theta_target + accept_deg,
        color="green", alpha=0.12,
        label=f"±{accept_deg:.0f}° accept band",
    )
    for b_t in state.get("source_boundaries_s", []):
        ax2.axvline(b_t, color="purple", lw=0.7, ls=":", alpha=0.8)
    if state["null_first_t"] is not None:
        ax2.axvline(state["null_first_t"], color="red", lw=1.0, ls="--")
    ax2.set_xlabel("time (s, concatenated sources)")
    ax2.set_ylabel("θ̂ (deg)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=7, ncol=2)
    # Pin θ̂ axis to the full active hemisphere [-90°, +90°] across all four
    # figures — the ±5° accept band then renders as the thin slice it really
    # is, so the reader can see the trace landing precisely on θ_target out
    # of the whole search space (instead of the auto-zoom that made every
    # figure look like a flat line at the target value).
    ax2.set_ylim(-90.0, 90.0)
    ax2.set_yticks(np.arange(-90.0, 90.0001, 30.0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parent / "config.yaml")
    args = p.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    sys.exit(main())
