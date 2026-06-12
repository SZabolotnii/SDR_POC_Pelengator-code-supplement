#!/usr/bin/env python3
"""Write a reproducibility manifest for the POC pelengator artifacts.

Collects git SHA, environment, and content hashes of every result JSON, figure,
and analysis script behind the HAIT manuscript, so the figures/tables can be
traced to an exact commit (arXiv ancillary-file convention, see CLAUDE.md).

Output: experiments/poc_pelengator/results/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results"
FIGURES = REPO_ROOT / "manuscript" / "hait_poc_pelengator" / "figures"
MANIFEST = RESULTS / "manifest.json"

RESULT_JSONS = [
    "a3_fingerprint_report.json", "a3_ablation_report.json",
    "a3_ablation_nonlinear_report.json", "nonlinear_gate_report.json",
    "a4_acceptance_report.json",
    "a5_proxy_report.json", "conditioning.json", "mismatch_sensitivity.json",
    "multipath_sensitivity.json", "field_prior.json",
]
# The formatted HAIT manuscripts embed the JPG figures (HAIT requires JPG), so
# the reproducibility trail must hash those exact submitted artifacts. The PNG
# masters are kept separately as source provenance.
FIGURE_STEMS = [
    "pipeline_diagram", "a5_summary", "conditioning",
    "mismatch_sensitivity", "multipath_sensitivity", "field_prior",
    "nonlinear_gate",
]
FIGURE_JPGS = [f"{s}.jpg" for s in FIGURE_STEMS]   # embedded in the submission
SOURCE_FIGURE_PNGS = [f"{s}.png" for s in FIGURE_STEMS]  # plotting masters
SCRIPTS = [
    "train_fingerprint.py", "run_a4_acceptance.py", "run_proxy.py",
    "analyze_conditioning.py", "analyze_mismatch.py", "run_multipath.py",
    "analyze_field_prior.py", "plot_a5_summary.py", "ablate_fingerprint.py",
    "ablate_fingerprint_nonlinear.py", "run_nonlinear_gate.py",
]


def _sha256(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    data = path.read_bytes()
    return {"present": True, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()[:16]}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "result_jsons": {n: _sha256(RESULTS / n) for n in RESULT_JSONS},
        "figures": {n: _sha256(FIGURES / n) for n in FIGURE_JPGS},
        "source_figures": {n: _sha256(FIGURES / n) for n in SOURCE_FIGURE_PNGS},
        "scripts": {n: _sha256(HERE / n) for n in SCRIPTS},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    missing = [n for group in ("result_jsons", "figures", "source_figures", "scripts")
               for n, v in manifest[group].items() if not v["present"]]
    print(f"wrote {MANIFEST}  (git {manifest['git_sha'][:8]}, "
          f"dirty={manifest['git_dirty']})")
    if missing:
        print(f"  WARNING missing: {missing}")
    else:
        print(f"  all {len(RESULT_JSONS)} JSONs + {len(FIGURE_JPGS)} JPG figures "
              f"(+{len(SOURCE_FIGURE_PNGS)} PNG sources) + {len(SCRIPTS)} scripts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
