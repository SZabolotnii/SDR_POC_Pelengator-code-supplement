# Code supplement — Software concept of a dual-channel SDR direction finder for DJI Mavic signals

Verification code supplement for the manuscript

> *"Software concept of a dual-channel SDR direction finder for DJI Mavic
> signals: a quantitative error budget based on open data"*,
> submitted to **Herald of Advanced Information Technology**.

This repository contains everything needed to verify every numerical result
reported in the paper: the processing-chain source code, the experiment
drivers, the exact configuration, the trained fingerprint model, and the
reference result artifacts (JSON) that the paper's tables and figures are
generated from.

## Repository layout

```
src/sdr_kunchenko/            processing-chain library
  rf/packet_detector.py         burst detection on the I/Q stream
  rf/dual_channel_simulator.py  cos² beam geometry, AWGN/multipath dual-channel proxy
  rf/fingerprint.py             DSGE + log-PSD "Mavic vs rest" gating filter
  doa/differential.py           per-burst amplitude difference Δ_dB
  doa/averaging.py              rolling median + Newey–West (HAC) standard error
  doa/null_detector.py          null event detector + theta_from_delta_db inverse
vendor/dsge-toolkit/scripts/  vendored frac-DSGE feature extractor (same code as Phase-6 pipeline)
experiments/poc_pelengator/   experiment drivers, config.yaml, trained model,
                              reference results (results/*.json)
manuscript/.../figures/       reference figures as used in the paper
tests/                        unit tests for the pipeline modules
data/tampere_extract/         place the open Tampere/Zenodo recordings here (see its README)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q          # smoke check of the pipeline modules (no RF data needed)
```

Python ≥ 3.10. Dependencies: numpy, scipy, scikit-learn, cloudpickle,
matplotlib, pyyaml.

**Data.** Reference artifacts in `experiments/poc_pelengator/results/` let you
check every number in the paper without downloading anything. To *re-run* the
experiments, download the open Tampere recordings (Zenodo
[10.5281/zenodo.4264467](https://doi.org/10.5281/zenodo.4264467)) — see
[`data/tampere_extract/README.md`](data/tampere_extract/README.md).

## Paper claim → script → artifact map

All scripts are run from the repository root, e.g.
`python experiments/poc_pelengator/train_fingerprint.py`.

| Paper item | Claim / numbers | Script | Reference artifact |
|---|---|---|---|
| Table 3 | fingerprint precision 0.9514, recall 0.9133, ROC AUC 0.9818, threshold 0.051, 3500/1500 windows, 19 features | `train_fingerprint.py` | `results/a3_fingerprint_report.json` |
| Ablation (Results, ¶1) | PSD-only AUC 0.9840 vs full 0.9818; DSGE-only 0.4814 (linear) / 0.74 RF, 0.72 SVM (nonlinear); RF full 0.991 vs PSD 0.983 | `ablate_fingerprint.py`, `ablate_fingerprint_nonlinear.py` | `results/a3_ablation_report.json`, `results/a3_ablation_nonlinear_report.json` |
| Table 4, Fig. 2 | end-to-end proxy: 279 packets → 123 kept, 4/4 scenarios within 5°, Δ_dB and HAC SE per angle, ≈60 s runtime | `run_proxy.py` (+ `plot_a5_summary.py`) | `results/a5_proxy_report.json`, `results/a5_proxy_log.jsonl` |
| Synthetic check (Results, ¶3) | 5/5 angles, max error 0.70° at +30° boresight, SNR 25 dB | `run_a4_acceptance.py` | `results/a4_acceptance_report.json` |
| Table 5, Fig. 3 | conditioning: well-conditioned sector ±30°, slopes 0.094/0.455/≈0/0.227 dB/°, σ_θ 10.6°/2.2°/∞/4.4° at 1 dB SE | `analyze_conditioning.py` | `results/conditioning.json` |
| Fig. 4 | calibration mismatch: 1 dB gain imbalance → 3.48° mean error (well-conditioned) / 8.10° (saturated, max 60.4°); 2° boresight → ≈2° bias; analytic vs simulated Δ_dB ≤ 0.0019 dB | `analyze_mismatch.py` | `results/mismatch_sensitivity.json` |
| Table 5, Fig. 5 | multipath (2-ray, Rayleigh): θ=0° stays within 2.7–7.4°, saturated angles reach 17–50° | `run_multipath.py` | `results/multipath_sensitivity.json` |
| Fig. 6 | field prior π=1%: precision 0.95 → 0.46; threshold recalibration → empirical FPR 0 on 1200 negatives; rule of three ⇒ precision ≳ 0.79 | `analyze_field_prior.py` | `results/field_prior.json` |
| Table 7, Fig. 7 | LR 0.9825±0.0031 vs RF 0.9880±0.0041 vs SVM-RBF 0.9614±0.0093 (5 seeds); recall at FPR=0: 0.91/0.91/0.90; shared 1%-prior precision LCB 0.75 (Clopper–Pearson, 0/1200) | `run_nonlinear_gate.py` | `results/nonlinear_gate_report.json` |
| Table 2 | end-to-end proxy configuration | — | `experiments/poc_pelengator/config.yaml` |
| Provenance | git SHA, environment, content hashes of all artifacts | `write_manifest.py` | `results/manifest.json` |

Definitions referenced in the paper:

- `cosine_beam_pattern` — `src/sdr_kunchenko/rf/dual_channel_simulator.py`:
  power-domain gain G(θ) = cos²((θ−φ)/(BW₃/2)·π/4), clamped at the −60 dB
  pattern floor; the inter-beam level difference is
  Δ_dB(θ) = 10·log₁₀(G_a(θ)/G_b(θ)).
- `theta_from_delta_db` — `src/sdr_kunchenko/doa/null_detector.py`:
  numerical inverse θ̂ = Δ_dB⁻¹(Δ) via 1-D grid search restricted to the
  active (non-floored) sector.

## Reproducing the experiments

With the Tampere recordings in place (see `data/tampere_extract/README.md`):

```bash
# 1. Train the Mavic-vs-rest fingerprint gate (Table 3)
python experiments/poc_pelengator/train_fingerprint.py

# 2. Feature-group ablation (linear and nonlinear)
python experiments/poc_pelengator/ablate_fingerprint.py
python experiments/poc_pelengator/ablate_fingerprint_nonlinear.py

# 3. End-to-end proxy, 4 angular scenarios (Table 4, Fig. 2)
python experiments/poc_pelengator/run_proxy.py
python experiments/poc_pelengator/plot_a5_summary.py

# 4. Controlled synthetic acceptance (5 angles, SNR 25 dB)
python experiments/poc_pelengator/run_a4_acceptance.py

# 5. Error-budget analyses (Tables 5–7, Figs. 3–7)
python experiments/poc_pelengator/analyze_conditioning.py
python experiments/poc_pelengator/analyze_mismatch.py
python experiments/poc_pelengator/run_multipath.py
python experiments/poc_pelengator/analyze_field_prior.py
python experiments/poc_pelengator/run_nonlinear_gate.py
```

Steps 4–5 of the error-budget analyses (`analyze_conditioning.py`,
`analyze_mismatch.py`) need no RF data at all — they evaluate the analytic
beam geometry. `run_multipath.py`, `analyze_field_prior.py` and
`run_nonlinear_gate.py` reuse the detector/fingerprint outputs and the
Tampere recordings.

All randomness is seeded (`seed: 42` in `config.yaml` and in the scripts;
the nonlinear-gate study sweeps seeds 42–46), so re-runs reproduce the
reference artifacts up to floating-point/platform noise.

## Relation to the research repository

This supplement is a frozen, self-contained extract of the author's research
repository (`sdr-kunchenko`, Phase 7 / Stream A). Code is byte-identical to
the research repository, with one documented exception:
`src/sdr_kunchenko/rf/fingerprint.py` resolves the frac-DSGE feature
extractor from the vendored copy in `vendor/dsge-toolkit/scripts/` instead of
an external checkout (marked with a "Code-supplement note" comment at the
constant `DEFAULT_DSGE_TOOLKIT`).

`write_manifest.py` records provenance against the research repository's
manuscript tree and is included for completeness; it is not required for
verification.

## License and citation

MIT License (see `LICENSE`). Dataset: Tampere University drone RF recordings,
Zenodo DOI [10.5281/zenodo.4264467](https://doi.org/10.5281/zenodo.4264467)
(credit the original authors when reusing the data).
