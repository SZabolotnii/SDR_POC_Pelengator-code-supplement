#!/usr/bin/env python3
"""Feature-group ablation for the A3 Mavic-vs-rest fingerprint filter.

Question: do the four fractional DSGE (Kunchenko-space) Re/Im descriptors
actually pull weight, or does the 15-bin log-PSD carry the classifier?

Reuses the exact A3 data pipeline (loader, per-label time split, seed) from
``train_fingerprint.py`` so the comparison is apples-to-apples, builds the
DSGE and PSD feature blocks once, then trains three LogisticRegression models
on the same train/test split:

  * full   — DSGE ⊕ PSD            (the deployed A3 vector)
  * psd    — PSD only              (drop Kunchenko features)
  * dsge   — DSGE only             (Kunchenko features alone)

Reports ROC AUC (threshold-free) plus the §A3 tuned precision/recall for each.

Output: experiments/poc_pelengator/results/a3_ablation_report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from train_fingerprint import (  # same dir
    SEED,
    TARGET_PRECISION,
    TARGET_RECALL,
    build_dataset,
    per_label_time_split,
    tune_threshold,
)

from sdr_kunchenko.rf.fingerprint import (
    FingerprintConfig,
    _import_complex_dsge,
    spectral_log_psd_features,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _fit_eval(feat_tr, y_tr, feat_te, y_te, cfg):
    scaler = StandardScaler().fit(feat_tr)
    clf = LogisticRegression(max_iter=cfg.lr_max_iter, C=cfg.lr_C).fit(
        scaler.transform(feat_tr), y_tr
    )
    proba = clf.predict_proba(scaler.transform(feat_te))[:, 1]
    auc = float(roc_auc_score(y_te, proba))
    tune = tune_threshold(
        y_te, proba, target_precision=TARGET_PRECISION, target_recall=TARGET_RECALL
    )
    return {
        "n_features": int(feat_tr.shape[1]),
        "test_roc_auc": auc,
        "tuned_threshold": tune["threshold"],
        "precision": tune["precision"],
        "recall": tune["recall"],
        "verdict": "PASS" if tune["accepts"] else "FAIL",
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("=== Loading Tampere 2.4 GHz dataset (A3 pipeline) ===", flush=True)
    x_all, y, t, _ = build_dataset(rng)
    tr_idx, te_idx = per_label_time_split(y, t, 0.7)
    x_tr, y_tr = x_all[tr_idx], y[tr_idx]
    x_te, y_te = x_all[te_idx], y[te_idx]
    print(f"Train {x_tr.shape} / Test {x_te.shape}", flush=True)

    cfg = FingerprintConfig(window_size=x_all.shape[1])

    # Fit the DSGE extractor on the training split only (same as production).
    print("=== Fitting DSGE extractor + building feature blocks ===", flush=True)
    dsge_cls = _import_complex_dsge(None)
    t0 = time.time()
    dsge = dsge_cls(mode=cfg.dsge_mode, basis=cfg.dsge_basis, n=cfg.dsge_n,
               ridge=cfg.dsge_ridge).fit(x_tr, y_tr)
    dsge_tr, dsge_te = dsge.transform(x_tr), dsge.transform(x_te)
    psd_tr = spectral_log_psd_features(x_tr, n_bins=cfg.n_spectral_bins)
    psd_te = spectral_log_psd_features(x_te, n_bins=cfg.n_spectral_bins)
    print(f"  DSGE dim={dsge_tr.shape[1]}, PSD dim={psd_tr.shape[1]} "
          f"({time.time()-t0:.1f}s)", flush=True)

    groups = {
        "full": (np.hstack([dsge_tr, psd_tr]), np.hstack([dsge_te, psd_te])),
        "psd_only": (psd_tr, psd_te),
        "dsge_only": (dsge_tr, dsge_te),
    }
    report = {
        "seed": SEED, "dsge_dim": int(dsge_tr.shape[1]),
        "psd_dim": int(psd_tr.shape[1]), "groups": {},
    }
    print("\n=== Ablation results ===", flush=True)
    for name, (ftr, fte) in groups.items():
        res = _fit_eval(ftr, y_tr, fte, y_te, cfg)
        report["groups"][name] = res
        print(f"  {name:<10} feat={res['n_features']:>2}  AUC={res['test_roc_auc']:.4f}"
              f"  P={res['precision']:.4f}  R={res['recall']:.4f}  {res['verdict']}",
              flush=True)

    full = report["groups"]["full"]["test_roc_auc"]
    psd = report["groups"]["psd_only"]["test_roc_auc"]
    report["dsge_auc_gain_over_psd"] = round(full - psd, 4)
    print(f"\n  ΔAUC(full − psd_only) = {full - psd:+.4f}", flush=True)

    out = RESULTS_DIR / "a3_ablation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  Saved → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
