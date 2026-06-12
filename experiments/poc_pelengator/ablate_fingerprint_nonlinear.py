#!/usr/bin/env python3
"""Nonlinear-probe ablation: is there hidden nonlinear signal in the DSGE block?

The linear (LogisticRegression) ablation in ``ablate_fingerprint.py`` found the
four fractional DSGE (Kunchenko-space) Re/Im descriptors at chance for
Mavic-vs-rest (AUC 0.48). ROC AUC under a linear model only measures *linear*
separability, so this script re-tests each feature block under nonlinear models
(RandomForest, SVM-rbf) alongside the LogisticRegression baseline, on the same
A3 train/test split. If DSGE-only stays near 0.5 under RF and SVM-rbf, there is
no hidden nonlinear signal; a jump would indicate the linear probe missed it.

Output: experiments/poc_pelengator/results/a3_ablation_nonlinear_report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from train_fingerprint import (  # same dir
    SEED,
    build_dataset,
    per_label_time_split,
)

from sdr_kunchenko.rf.fingerprint import (
    FingerprintConfig,
    _import_complex_dsge,
    spectral_log_psd_features,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _classifiers():
    """Fresh classifier factories (standardized inputs where it matters)."""
    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=SEED),
        "svm_rbf": lambda: make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    }


def _auc(model_name, factory, x_tr, y_tr, x_te, y_te) -> float:
    clf = factory()
    clf.fit(x_tr, y_tr)
    if model_name == "svm_rbf":          # use the margin, no probability calib
        scores = clf.decision_function(x_te)
    else:
        scores = clf.predict_proba(x_te)[:, 1]
    return float(roc_auc_score(y_te, scores))


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("=== Loading Tampere 2.4 GHz dataset (A3 pipeline) ===", flush=True)
    x_all, y, t, _ = build_dataset(rng)
    tr_idx, te_idx = per_label_time_split(y, t, 0.7)
    x_tr, y_tr = x_all[tr_idx], y[tr_idx]
    x_te, y_te = x_all[te_idx], y[te_idx]

    cfg = FingerprintConfig(window_size=x_all.shape[1])
    dsge_cls = _import_complex_dsge(None)
    dsge = dsge_cls(mode=cfg.dsge_mode, basis=cfg.dsge_basis, n=cfg.dsge_n,
                    ridge=cfg.dsge_ridge).fit(x_tr, y_tr)
    dsge_tr, dsge_te = dsge.transform(x_tr), dsge.transform(x_te)
    psd_tr = spectral_log_psd_features(x_tr, n_bins=cfg.n_spectral_bins)
    psd_te = spectral_log_psd_features(x_te, n_bins=cfg.n_spectral_bins)

    blocks = {
        "dsge_only": (dsge_tr, dsge_te),
        "psd_only": (psd_tr, psd_te),
        "full": (np.hstack([dsge_tr, psd_tr]), np.hstack([dsge_te, psd_te])),
    }
    clfs = _classifiers()
    report = {"seed": SEED, "dsge_dim": int(dsge_tr.shape[1]),
              "psd_dim": int(psd_tr.shape[1]), "metric": "test_roc_auc",
              "results": {}}

    print("\n=== ROC AUC: feature-block x classifier ===", flush=True)
    header = f"{'block':<10} " + " ".join(f"{c:>14}" for c in clfs)
    print(header, flush=True)
    for bname, (ftr, fte) in blocks.items():
        row = {}
        for cname, factory in clfs.items():
            t0 = time.time()
            row[cname] = _auc(cname, factory, ftr, y_tr, fte, y_te)
            row[f"{cname}_sec"] = round(time.time() - t0, 1)
        report["results"][bname] = row
        print(f"{bname:<10} " + " ".join(f"{row[c]:>14.4f}" for c in clfs),
              flush=True)

    d = report["results"]["dsge_only"]
    best = max(d[c] for c in clfs)
    report["dsge_only_best_auc"] = round(best, 4)
    report["dsge_only_nonlinear_signal"] = bool(best >= 0.60)
    print(f"\n  DSGE-only best AUC across classifiers = {best:.4f} "
          f"-> nonlinear signal: {'YES' if best >= 0.60 else 'no (≈chance)'}",
          flush=True)

    out = RESULTS_DIR / "a3_ablation_nonlinear_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  Saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
