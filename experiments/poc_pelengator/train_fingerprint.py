#!/usr/bin/env python3
"""Train Mavic-vs-rest FingerprintFilter on Tampere 2.4 GHz drones.

Phase 7 / A3 (Plan-B primary path after A1 ruled out OcuSync-1 header
extraction). Reproduces the Phase 6 hybrid pipeline (frac DSGE Re/Im +
log-PSD + LogisticRegression) packaged behind FingerprintFilter,
trained as a binary classifier (target = Mavic Pro vs everything else).

Acceptance (plan §A3):  precision ≥ 0.95  AND  recall ≥ 0.85  on hold-out.

Outputs
-------
  experiments/poc_pelengator/models/mavic_pro_fingerprint.pkl
  experiments/poc_pelengator/results/a3_fingerprint_report.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from sdr_kunchenko.rf.fingerprint import FingerprintConfig, FingerprintFilter

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tampere_extract"
OUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"

# label = 1 → target (Mavic Pro). label = 0 → everything else.
CLASSES = [
    ("DJI_Mavic_Pro",    DATA_DIR / "DJI_mavic_pro_2G.bin",        1),
    ("DJI_Inspire_2",    DATA_DIR / "DJI_inspire_2_2G.bin",        0),
    ("DJI_Phantom_4",    DATA_DIR / "DJI_phantom_4_2G.bin",        0),
    ("Parrot_Disco",     DATA_DIR / "Parrot_disco_2G.bin",         0),
    ("Yuneec_Typhoon_H", DATA_DIR / "Yuneec_typhoon_h_2G_1of2.bin", 0),
]

WINDOW = 4096
N_PER_FILE = 1000          # subsample windows per recording
TRAIN_FRAC = 0.7
SEED = 42
TARGET_PRECISION = 0.95
TARGET_RECALL = 0.85


# ---------------------------------------------------------------- loader


def load_burst_windows(
    path: Path, n_windows: int, win: int, rng: np.random.Generator,
    *, energy_pct: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_windows` per-window-z-scored complex64 chunks from a .bin.

    Mirrors the Phase 6 loader: read 2× headroom of int16 IQ, slice into
    fixed-size windows, keep an energy-thresholded random subsample
    (so silence between bursts does not dominate), then per-window z-score.
    Returns (windows, sample-offsets) — offsets feed the time-ordered split.
    """
    headroom = 2
    n_int16 = 2 * n_windows * win * headroom
    data = np.fromfile(path, dtype="<i2", count=n_int16)
    if data.size < 2 * win:
        raise RuntimeError(f"{path.name} too short: only {data.size} int16")
    iq = data.astype(np.float32).reshape(-1, 2)
    sig = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    nc = len(sig) // win
    chunks = sig[: nc * win].reshape(nc, win)

    energies = (chunks.real ** 2 + chunks.imag ** 2).mean(axis=1)
    threshold = float(np.percentile(energies, energy_pct))
    high = np.where(energies >= threshold)[0]
    if high.size >= n_windows:
        chosen = rng.choice(high, size=n_windows, replace=False)
    else:
        low = np.where(energies < threshold)[0]
        extra = rng.choice(low, size=n_windows - high.size, replace=False)
        chosen = np.concatenate([high, extra])
    chosen = np.sort(chosen)
    windows = chunks[chosen]

    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-9
    windows = ((windows - mean) / std).astype(np.complex64)
    return windows, (chosen * win).astype(np.int64)


def build_dataset(rng: np.random.Generator):
    """Load all classes; return (X, y, t, per_class_summary)."""
    X_parts, y_parts, t_parts, summary = [], [], [], []
    for name, path, label in CLASSES:
        if not path.exists():
            print(f"  [skip] {name} — not at {path}", flush=True)
            summary.append({"name": name, "loaded": False})
            continue
        t0 = time.time()
        wins, ts = load_burst_windows(path, N_PER_FILE, WINDOW, rng)
        elapsed = time.time() - t0
        X_parts.append(wins)
        y_parts.append(np.full(wins.shape[0], label, dtype=int))
        t_parts.append(ts)
        summary.append({
            "name": name, "loaded": True, "label": int(label),
            "n_windows": int(wins.shape[0]),
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  {name:<18} label={label} windows={wins.shape[0]} "
              f"({elapsed:.1f}s)", flush=True)
    return (np.vstack(X_parts),
            np.concatenate(y_parts),
            np.concatenate(t_parts),
            summary)


def per_label_time_split(
    y: np.ndarray, t: np.ndarray, train_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-order each label, take first `train_frac` for train.

    Splitting per-label preserves class balance and prevents temporal
    leakage inside a class (whole-recording confounders).
    """
    train_idx, test_idx = [], []
    for c in np.unique(y):
        m = np.where(y == c)[0]
        order = m[np.argsort(t[m])]
        cut = int(train_frac * len(order))
        train_idx.append(order[:cut])
        test_idx.append(order[cut:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def tune_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, *,
    target_precision: float, target_recall: float,
) -> dict:
    """Pick the highest-recall threshold whose precision still meets target.

    Returns a dict with the tuned threshold, achieved precision/recall, and
    whether both acceptance bounds are satisfied. Falls back to "best
    precision" if no threshold reaches `target_precision`.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve appends (1, 0) — drop the trailing slot
    qualify = prec[:-1] >= target_precision
    if not qualify.any():
        idx = int(np.argmax(prec[:-1]))
        out = {
            "threshold": float(thr[idx]),
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "qualifies_precision": False,
            "qualifies_recall": False,
        }
    else:
        rec_q = np.where(qualify, rec[:-1], -np.inf)
        idx = int(np.argmax(rec_q))
        out = {
            "threshold": float(thr[idx]),
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "qualifies_precision": True,
            "qualifies_recall": bool(rec[idx] >= target_recall),
        }
    out["accepts"] = (
        out["precision"] >= target_precision and out["recall"] >= target_recall
    )
    return out


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print("=== Loading Tampere 2.4 GHz dataset ===", flush=True)
    X, y, t, class_summary = build_dataset(rng)
    print(f"\nTotal windows: {X.shape}; positives: {(y == 1).sum()}, "
          f"negatives: {(y == 0).sum()}", flush=True)

    tr_idx, te_idx = per_label_time_split(y, t, TRAIN_FRAC)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f"Train: {X_tr.shape} ({(y_tr == 1).sum()} pos / {(y_tr == 0).sum()} neg)",
           flush=True)
    print(f"Test:  {X_te.shape} ({(y_te == 1).sum()} pos / {(y_te == 0).sum()} neg)",
           flush=True)

    print("\n=== Fitting FingerprintFilter (frac-DSGE-re_im + log-PSD + LR) ===",
           flush=True)
    cfg = FingerprintConfig(window_size=WINDOW)
    t0 = time.time()
    ff = FingerprintFilter(config=cfg).fit(X_tr, y_tr)
    fit_s = time.time() - t0
    md = ff.fit_metadata()
    print(f"  fit time: {fit_s:.1f}s; "
          f"feature_dim={md['feature_dim']} (DSGE={md['dsge_dim']}, "
          f"PSD={md['psd_dim']})", flush=True)

    print("\n=== Hold-out scoring ===", flush=True)
    t0 = time.time()
    y_proba = ff.predict_proba(X_te)
    pred_s = time.time() - t0
    auc = float(roc_auc_score(y_te, y_proba))
    print(f"  predict time: {pred_s:.1f}s ; ROC AUC = {auc:.4f}", flush=True)

    tune = tune_threshold(
        y_te, y_proba,
        target_precision=TARGET_PRECISION, target_recall=TARGET_RECALL,
    )
    ff.set_threshold(tune["threshold"])
    verdict = "PASS" if tune["accepts"] else "FAIL"
    print(
        f"  threshold = {tune['threshold']:.4f} → "
        f"precision = {tune['precision']:.4f}, recall = {tune['recall']:.4f}",
        flush=True,
    )
    print(
        f"  ACCEPTANCE: {verdict} "
        f"(need precision ≥ {TARGET_PRECISION}, recall ≥ {TARGET_RECALL})",
        flush=True,
    )

    model_path = MODEL_DIR / "mavic_pro_fingerprint.pkl"
    ff.save(model_path)
    print(f"\n  Saved model → {model_path}", flush=True)

    report = {
        "config": asdict(cfg),
        "classes": class_summary,
        "n_train_windows": int(len(y_tr)),
        "n_test_windows": int(len(y_te)),
        "n_train_pos": int((y_tr == 1).sum()),
        "n_test_pos": int((y_te == 1).sum()),
        "fit_metadata": md,
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(pred_s, 2),
        "test_roc_auc": auc,
        "target_precision": TARGET_PRECISION,
        "target_recall": TARGET_RECALL,
        "tuned": tune,
        "verdict": verdict,
        "model_path": str(model_path.relative_to(Path.cwd())) if str(model_path).startswith(str(Path.cwd())) else str(model_path),
        "seed": SEED,
    }
    report_path = RESULTS_DIR / "a3_fingerprint_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Saved report → {report_path}", flush=True)

    return 0 if tune["accepts"] else 1


if __name__ == "__main__":
    sys.exit(main())
