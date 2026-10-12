"""Probability-average ensemble of finished runs.

Writes runs/<out>/{oof,test_probs}/*.npy (uint8) + config.yaml so that `pipeline.py tune`,
`scorer.py` and `scorer.py --predict` work on the ensemble exactly as on a single run.
Usage: python ensemble.py --out <run_id> --runs <run_a> <run_b> [--weights 0.5 0.5]
"""
import argparse
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.runs), np.float32)
    w /= w.sum()
    src = [os.path.join(HERE, "runs", r) for r in a.runs]
    out = os.path.join(HERE, "runs", a.out)
    for sub in ("oof", "test_probs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        names = sorted(f for f in os.listdir(os.path.join(src[0], sub)) if f.endswith(".npy"))
        for f in names:
            acc = sum(wi * np.load(os.path.join(s, sub, f)).astype(np.float32) for wi, s in zip(w, src))
            np.save(os.path.join(out, sub, f), np.round(acc).astype(np.uint8))
        print(sub, len(names))
    for k in range(5):
        open(os.path.join(out, "oof", f"done_f{k}"), "w").close()
    with open(os.path.join(src[0], "config.yaml"), encoding="utf-8") as f:
        cfg = f.read()
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg.replace(cfg.split("\n")[0], f"run_id: {a.out}", 1))
        f.write(f"\nensemble:\n  members: {a.runs}\n  weights: {w.round(3).tolist()}\n")
    with open(os.path.join(out, "attribution.md"), "w", encoding="utf-8") as f:
        f.write(f"# Attribution — {a.out}\n\nProbability-average ensemble of {a.runs} (weights {w.round(3).tolist()}).\n"
                "Ideas and citations: union of members' attribution.md.\n"
                "- ensemble: dense-unet-ensemble — tushhaaarrrr/solar-filament-segmentation-2026-0-40\n")


if __name__ == "__main__":
    main()
