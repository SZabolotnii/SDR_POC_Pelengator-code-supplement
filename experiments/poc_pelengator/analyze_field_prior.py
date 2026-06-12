#!/usr/bin/env python3
"""Base-rate (field-prior) correction for the Mavic-vs-rest fingerprint gate.

Closes the "fingerprint base-rate shift" limitation. The A3 metrics
(precision 0.9514) were measured on a test set whose Mavic base rate is
300/1500 = 20 %. In the open air the target base rate may be ~1 %, where the
same threshold yields far lower precision. Class-conditional detection rates
TPR(t)=P(score≥t|Mavic) and FPR(t)=P(score≥t|rest) are base-rate-independent
properties of the classifier, so the Bayesian precision at any field prior π is

    precision(t; π) = π·TPR(t) / [ π·TPR(t) + (1-π)·FPR(t) ] .

We reproduce the deterministic A3 split (seed 42), re-score the held-out set,
and report (a) precision vs π at the operating thresholds and (b) the
recalibrated threshold that restores target precision at π = 1 %, with its
recall cost.

Outputs:
  experiments/poc_pelengator/results/field_prior.json
  manuscript/hait_poc_pelengator/figures/field_prior.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results" / "field_prior.json"
FIGURE = REPO_ROOT / "manuscript" / "hait_poc_pelengator" / "figures" / "field_prior.png"

sys.path.insert(0, str(HERE))
from train_fingerprint import (  # noqa: E402
    MODEL_DIR,
    SEED,
    TRAIN_FRAC,
    build_dataset,
    per_label_time_split,
)

from sdr_kunchenko.rf.fingerprint import FingerprintFilter  # noqa: E402

PRIORS = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
THR_A3 = 0.0510          # A3 tuned threshold
THR_A5 = 0.5             # A5 proxy demo threshold
TARGET_PRECISION = 0.95
FIELD_PRIOR = 0.01


def precision_at_prior(tpr: np.ndarray, fpr: np.ndarray, prior: float) -> np.ndarray:
    num = prior * tpr
    den = prior * tpr + (1.0 - prior) * fpr
    out = np.full_like(tpr, np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    # When no positives and no false positives survive, precision is undefined;
    # treat a gate that admits nothing as precision 1 (vacuously) only if tpr>0.
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    print("=== reproducing A3 split (seed 42) ===", flush=True)
    X, y, t, _ = build_dataset(rng)  # noqa: N806  (ML feature matrix)
    _, te_idx = per_label_time_split(y, t, TRAIN_FRAC)
    X_te, y_te = X[te_idx], y[te_idx]  # noqa: N806
    print(f"  test: {X_te.shape}  ({(y_te == 1).sum()} pos / {(y_te == 0).sum()} neg)",
          flush=True)

    ff = FingerprintFilter.load(MODEL_DIR / "mavic_pro_fingerprint.pkl")
    scores = ff.predict_proba(X_te)
    pos = scores[y_te == 1]
    neg = scores[y_te == 0]
    test_prior = float((y_te == 1).mean())
    print(f"  test base rate = {test_prior:.3f}; ROC AUC check = "
          f"{_auc(pos, neg):.4f}", flush=True)

    thr = np.linspace(0.0, 1.0, 2001)
    tpr = np.array([(pos >= t).mean() for t in thr])
    fpr = np.array([(neg >= t).mean() for t in thr])

    def metrics_at(t_val: float, prior: float) -> dict:
        i = int(np.argmin(np.abs(thr - t_val)))
        prec = precision_at_prior(tpr, fpr, prior)[i]
        return {"recall": round(float(tpr[i]), 4),
                "fpr": round(float(fpr[i]), 5),
                "precision": (None if np.isnan(prec) else round(float(prec), 4))}

    # precision vs prior at the two operating thresholds
    vs_prior = {}
    for pi in PRIORS:
        vs_prior[f"{pi:g}"] = {
            "thr_a3_0.051": metrics_at(THR_A3, pi),
            "thr_a5_0.5": metrics_at(THR_A5, pi),
        }

    # recalibrated threshold to restore TARGET_PRECISION at FIELD_PRIOR
    prec_field = precision_at_prior(tpr, fpr, FIELD_PRIOR)
    qualify = np.where(np.nan_to_num(prec_field, nan=0.0) >= TARGET_PRECISION)[0]
    if qualify.size:
        # smallest qualifying threshold = highest recall among precision-OK gates
        i_star = int(qualify[0])
        recal = {
            "field_prior": FIELD_PRIOR,
            "target_precision": TARGET_PRECISION,
            "recalibrated_threshold": round(float(thr[i_star]), 4),
            "recall_at_recal": round(float(tpr[i_star]), 4),
            "fpr_at_recal": round(float(fpr[i_star]), 6),
            "precision_at_recal": round(float(prec_field[i_star]), 4),
        }
    else:
        recal = {"field_prior": FIELD_PRIOR, "target_precision": TARGET_PRECISION,
                 "recalibrated_threshold": None,
                 "note": "target precision unreachable at this prior"}

    report = {
        "test_base_rate": round(test_prior, 4),
        "operating_thresholds": {"a3_tuned": THR_A3, "a5_demo": THR_A5},
        "precision_vs_prior": vs_prior,
        "recalibration": recal,
    }
    RESULTS.write_text(json.dumps(report, indent=2))
    print(f"saved → {RESULTS}", flush=True)
    print(f"  precision @ thr=0.051: π=0.20 → {vs_prior['0.2']['thr_a3_0.051']['precision']}, "
          f"π=0.01 → {vs_prior['0.01']['thr_a3_0.051']['precision']}", flush=True)
    print(f"  recalibration @ π=0.01: t*={recal.get('recalibrated_threshold')}, "
          f"recall={recal.get('recall_at_recal')}, "
          f"precision={recal.get('precision_at_recal')}", flush=True)

    # ---------------------------------------------------------------- figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    pri = np.array(PRIORS)
    for t_val, color, lab in ((THR_A3, "tab:blue", "A3 threshold 0.051"),
                              (THR_A5, "tab:orange", "A5 demo threshold 0.5")):
        i = int(np.argmin(np.abs(thr - t_val)))
        prec_curve = [precision_at_prior(tpr, fpr, p)[i] for p in pri]
        ax_a.semilogx(pri, prec_curve, marker="o", ms=4, color=color, label=lab)
    ax_a.axhline(TARGET_PRECISION, color="0.4", ls="--", lw=1.0,
                label=f"target precision {TARGET_PRECISION}")
    ax_a.axvline(FIELD_PRIOR, color="tab:red", ls=":", lw=1.0,
                label=f"field prior {FIELD_PRIOR:g}")
    ax_a.set_xlabel("Mavic base rate  π")
    ax_a.set_ylabel("precision")
    ax_a.set_title("Precision collapses at low field base rate")
    ax_a.set_ylim(0, 1.02)
    ax_a.grid(alpha=0.3, which="both")
    ax_a.legend(loc="lower right", fontsize=7.0)

    prec_f = precision_at_prior(tpr, fpr, FIELD_PRIOR)
    ax_b.plot(thr, np.nan_to_num(prec_f, nan=1.0), color="tab:purple",
             lw=1.6, label=f"precision (π={FIELD_PRIOR:g})")
    ax_b.plot(thr, tpr, color="tab:green", lw=1.6, label="recall")
    ax_b.axhline(TARGET_PRECISION, color="0.4", ls="--", lw=1.0)
    if recal.get("recalibrated_threshold") is not None:
        t_star = recal["recalibrated_threshold"]
        ax_b.axvline(t_star, color="tab:red", ls=":", lw=1.2,
                    label=f"recal. t*={t_star:g} (recall {recal['recall_at_recal']:.2f})")
    ax_b.axvline(THR_A3, color="tab:blue", ls=":", lw=0.9, alpha=0.7)
    ax_b.set_xlabel("threshold t")
    ax_b.set_ylabel("precision / recall")
    ax_b.set_title(f"Recalibration at π = {FIELD_PRIOR:g}")
    ax_b.set_ylim(0, 1.02)
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="center right", fontsize=7.0)

    fig.tight_layout()
    fig.savefig(FIGURE, dpi=300)
    plt.close(fig)
    print(f"saved → {FIGURE}", flush=True)
    return 0


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann–Whitney ROC AUC sanity check (no sklearn import needed)."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


if __name__ == "__main__":
    raise SystemExit(main())
