"""Re-run OOF inference of a finished run with a different TTA setting into a new run dir.

Copies model_f*.pt + config (with infer.tta overridden), regenerates oof/*.npy fold by fold.
Afterwards run `pipeline.py tune` and `pipeline.py predict` on the new dir as usual.
Usage: python reinfer.py --src <run_id> --dst <run_id> --tta 8
"""
import argparse
import os
import shutil
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import atomic_save_npy, build_model, load_cache, make_folds, predict_tta  # noqa: E402


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tta", type=int, default=8)
    a = ap.parse_args()
    src, dst = os.path.join(HERE, "runs", a.src), os.path.join(HERE, "runs", a.dst)
    os.makedirs(os.path.join(dst, "oof"), exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(src, "config.yaml"), encoding="utf-8"))
    cfg["run_id"], cfg["infer"]["tta"] = a.dst, a.tta
    cfg.setdefault("idea_keys", []).append(f"aug:dihedral-tta-{a.tta}")
    yaml.safe_dump(cfg, open(os.path.join(dst, "config.yaml"), "w", encoding="utf-8"), sort_keys=False)
    shutil.copy(os.path.join(src, "attribution.md"), os.path.join(dst, "attribution.md"))
    imgs, _, stems = load_cache(cfg["data"]["res"])
    folds = make_folds()
    fold_of = dict(zip(folds.stem, folds.fold))
    for fold in cfg["train"]["folds"]:
        done = os.path.join(dst, "oof", f"done_f{fold}")
        if os.path.exists(done):
            continue
        ck = os.path.join(dst, f"model_f{fold}.pt")
        if not os.path.exists(ck):
            shutil.copy(os.path.join(src, f"model_f{fold}.pt"), ck)
        m = build_model(dict(cfg, model=dict(cfg["model"], weights=None))).cuda()
        m.load_state_dict(torch.load(ck, map_location="cuda"))
        m.eval()
        for i, s in enumerate(stems):
            if fold_of[s] == fold:
                p = predict_tta(m, imgs[i], a.tta, "cuda")
                atomic_save_npy(os.path.join(dst, "oof", s + ".npy"), np.round(p * 255).astype(np.uint8))
        open(done, "w").close()
        print("fold", fold, "done", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
