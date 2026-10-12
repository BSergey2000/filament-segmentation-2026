"""Solar filament instance segmentation pipeline (own implementation).

Stages (python pipeline.py <cmd> --config runs/<run_id>/config.yaml):
  cache    build in-memory-friendly arrays at train resolution (images, soft annotator-mean targets)
  train    train every fold (skips folds whose checkpoint + oof already exist), writes OOF probs
  tune     grid-search post-processing on OOF with the host-exact PQ; writes cv_score.json
  predict  fold-ensemble + TTA inference on test, writes test_preds.csv (submission format)

Ideas re-implemented (see attribution.md of each run): GroupKFold by image, soft annotator-mean
target, U-Net + BCE/soft-Dice, connected-component instances, metric-aware instance keep threshold,
min-area filter, dihedral TTA.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.dirname(HERE)  # runs/<slug>
RAW = os.path.join(RUN_ROOT, "data", "raw", "MAGFiLO_1.0_Kaggle_2026")
PROC = os.path.join(RUN_ROOT, "data", "processed")
FULL = 2048
sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------- io helpers
def imread(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(**kw):
    with open(os.path.join(RUN_ROOT, "progress.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": now_utc(), "stage": "stage2", **kw}) + "\n")


def atomic_save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- data
def make_folds(n_folds=5, seed=42):
    path = os.path.join(HERE, "folds.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    idx = pd.read_csv(os.path.join(PROC, "index.csv"))
    stems = np.array(sorted(idx.stem.unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(stems)
    folds = pd.DataFrame({"stem": stems, "fold": np.arange(len(stems)) % n_folds}).sort_values("stem")
    folds.to_csv(path, index=False)
    return folds


def cache_paths(res):
    d = os.path.join(PROC, f"cache_{res}")
    return d, os.path.join(d, "images.npy"), os.path.join(d, "soft.npy"), os.path.join(d, "stems.csv")


def build_cache(res):
    d, p_img, p_soft, p_stems = cache_paths(res)
    if os.path.exists(p_soft):
        return
    os.makedirs(d, exist_ok=True)
    idx = pd.read_csv(os.path.join(PROC, "index.csv"))
    stems = sorted(idx.stem.unique())
    imgs = np.zeros((len(stems), res, res), np.uint8)
    soft = np.zeros((len(stems), res, res), np.uint8)
    for i, stem in enumerate(stems):
        img = imread(os.path.join(PROC, "images", stem + ".png"), cv2.IMREAD_GRAYSCALE)
        imgs[i] = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        acc = np.zeros((FULL, FULL), np.float32)
        sets = idx[idx.stem == stem].annot_set_id.tolist()
        for s in sets:
            acc += imread(os.path.join(PROC, "inst", s + ".png")) > 0
        acc /= len(sets)
        soft[i] = np.round(cv2.resize(acc, (res, res), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
        if i % 100 == 0:
            print("cache", i, len(stems), flush=True)
    np.save(p_img, imgs)
    np.save(p_soft, soft)
    pd.DataFrame({"stem": stems}).to_csv(p_stems, index=False)


def load_cache(res):
    _, p_img, p_soft, p_stems = cache_paths(res)
    return np.load(p_img), np.load(p_soft), pd.read_csv(p_stems).stem.tolist()


def gt_maps_for(stem, res, idx):
    """List of GT instance maps (one per annotator) at resolution res (nearest)."""
    out = []
    for s in idx[idx.stem == stem].annot_set_id:
        m = imread(os.path.join(PROC, "inst", s + ".png"))
        if res != FULL:
            m = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)
        out.append(m)
    return out


# ----------------------------------------------------------------------------- model
def build_model(cfg):
    import segmentation_models_pytorch as smp
    arch = getattr(smp, cfg["model"]["arch"])
    return arch(encoder_name=cfg["model"]["encoder"], encoder_weights=cfg["model"].get("weights", "imagenet"),
                in_channels=1, classes=1)


def soft_dice_loss(logits, target, eps=1.0):
    import torch
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = (p + target).sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def augment(img, tgt, rng, crop):
    h = img.shape[0]
    y, x = rng.randint(0, h - crop + 1, size=2)
    img, tgt = img[y:y + crop, x:x + crop], tgt[y:y + crop, x:x + crop]
    k = rng.randint(4)
    img, tgt = np.rot90(img, k), np.rot90(tgt, k)
    if rng.rand() < 0.5:
        img, tgt = img[:, ::-1], tgt[:, ::-1]
    img = img.astype(np.float32)
    if rng.rand() < 0.8:  # brightness / contrast / gamma jitter
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
        img = np.clip(img, 0, 255)
        img = 255.0 * (img / 255.0) ** rng.uniform(0.8, 1.25)
    return np.ascontiguousarray(img), np.ascontiguousarray(tgt)


def predict_tta(model, img_u8, tta, device):
    """img_u8: (H,W) uint8. Returns prob (H,W) float32 averaged over dihedral views."""
    import torch
    views = {1: [(0, False)], 2: [(0, False), (0, True)], 4: [(0, False), (0, True), (2, False), (2, True)],
             8: [(k, f) for k in range(4) for f in (False, True)]}[tta]
    acc = np.zeros(img_u8.shape, np.float32)
    x0 = img_u8.astype(np.float32) / 255.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for k, f in views:
            v = np.rot90(x0, k)
            v = v[:, ::-1] if f else v
            t = torch.from_numpy(np.ascontiguousarray(v))[None, None].to(device)
            p = torch.sigmoid(model(t)).float()[0, 0].cpu().numpy()
            p = p[:, ::-1] if f else p
            acc += np.rot90(p, -k)
    return acc / len(views)


def train(cfg, run_dir):
    import torch
    res, crop = cfg["data"]["res"], cfg["train"]["crop"]
    build_cache(res)
    imgs, soft, stems = load_cache(res)
    folds = make_folds()
    fold_of = dict(zip(folds.stem, folds.fold))
    fold_arr = np.array([fold_of[s] for s in stems])
    device = "cuda"
    os.makedirs(os.path.join(run_dir, "oof"), exist_ok=True)

    for fold in cfg["train"]["folds"]:
        ck = os.path.join(run_dir, f"model_f{fold}.pt")
        oof_done = os.path.join(run_dir, "oof", f"done_f{fold}")
        if os.path.exists(ck) and os.path.exists(oof_done):
            print(f"fold {fold} already done, skipping", flush=True)
            continue
        log_event(event="fold_started", run_id=cfg["run_id"], fold=fold)
        seed = cfg["train"]["seed"] + fold
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        tr = np.where(fold_arr != fold)[0]
        va = np.where(fold_arr == fold)[0]
        model = build_model(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
        epochs, bs = cfg["train"]["epochs"], cfg["train"]["batch"]
        steps_per_epoch = cfg["train"].get("samples_per_epoch", len(tr)) // bs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["train"]["lr"], total_steps=epochs * steps_per_epoch,
                                                    pct_start=0.05)
        scaler = torch.amp.GradScaler()
        bce = torch.nn.BCEWithLogitsLoss()
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for _ in range(steps_per_epoch):
                b = rng.choice(tr, bs)
                xs, ys = zip(*(augment(imgs[i], soft[i], rng, crop) for i in b))
                x = torch.from_numpy(np.stack(xs) / 255.0).float()[:, None].to(device)
                y = torch.from_numpy(np.stack(ys).astype(np.float32) / 255.0)[:, None].to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                loss = bce(logits.float(), y) + cfg["train"]["dice_w"] * soft_dice_loss(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
            print(f"fold {fold} ep {ep} loss {tot / steps_per_epoch:.4f} t {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), ck)
        # OOF probabilities (uint8) at train resolution
        model.eval()
        for i in va:
            p = predict_tta(model, imgs[i], cfg["infer"]["tta"], device)
            atomic_save_npy(os.path.join(run_dir, "oof", stems[i] + ".npy"), np.round(p * 255).astype(np.uint8))
        open(oof_done, "w").close()
        log_event(event="fold_done", run_id=cfg["run_id"], fold=fold, minutes=round((time.time() - t0) / 60, 1))
        del model
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- post-processing
def instances_from_prob(prob, thr, min_area=0, keep_thr=0.0):
    """prob (H,W) float in [0,1] -> (inst map int32, per-instance mean prob)."""
    lab, n = ndimage.label(prob > thr, structure=np.ones((3, 3)))
    if n == 0:
        return lab.astype(np.int32), np.zeros(0)
    ids = np.arange(1, n + 1)
    area = ndimage.sum_labels(np.ones_like(prob), lab, ids)
    mean = ndimage.mean(prob, lab, ids)
    keep = (area >= min_area) & (mean >= keep_thr)
    remap = np.zeros(n + 1, np.int32)
    remap[ids[keep]] = np.arange(1, keep.sum() + 1)
    return remap[lab], mean[keep]


def instance_stats(gt_maps, pred):
    """For pred instance map vs each GT map: per pred instance best IoU>0.5 match per annotator.
    Returns dict(area, n_gt per set, match: list over sets of (pred_id->(gt_id, iou)))"""
    n_pred = int(pred.max())
    matches = []
    for g in gt_maps:
        g = g.astype(np.int64)
        both = (g > 0) & (pred > 0)
        base = n_pred + 1
        keys, inter = np.unique(g[both] * base + pred[both], return_counts=True)
        ga = np.bincount(g.ravel(), minlength=int(g.max()) + 1)
        pa = np.bincount(pred.ravel(), minlength=base)
        m = {}
        for k, it in zip(keys.tolist(), inter.tolist()):
            gi, pi = divmod(k, base)
            iou = it / (ga[gi] + pa[pi] - it)
            if iou > 0.5:
                m[pi] = (gi, iou)
        matches.append((len(np.unique(g)) - 1, m))
    return matches


def tune(cfg, run_dir):
    idx = pd.read_csv(os.path.join(PROC, "index.csv"))
    folds = make_folds()
    res = cfg["data"]["res"]
    thrs = cfg["tune"]["thr"]
    min_areas = cfg["tune"]["min_area"]          # in FULL-res pixels
    keep_thrs = cfg["tune"]["keep_thr"]
    scale = (FULL / res) ** 2
    # records[thr] = list of (fold, area_full, mean, [match flags per set...]) aggregated as arrays
    agg = {}  # (thr, min_area, keep, fold) -> [sum_iou, tp, fp, fn]
    folds = folds[folds.fold.isin(cfg["train"]["folds"])]
    stems = folds.stem.tolist()
    t0 = time.time()
    for n, stem in enumerate(stems):
        fold = int(folds.loc[folds.stem == stem, "fold"].iloc[0])
        prob = np.load(os.path.join(run_dir, "oof", stem + ".npy")).astype(np.float32) / 255.0
        gts = gt_maps_for(stem, res, idx)
        for thr in thrs:
            pred, mean = instances_from_prob(prob, thr)
            area = np.bincount(pred.ravel(), minlength=int(pred.max()) + 1)[1:] * scale
            matches = instance_stats(gts, pred)
            for ma in min_areas:
                for kt in keep_thrs:
                    keep = (area >= ma) & (mean >= kt)
                    kept_ids = set((np.where(keep)[0] + 1).tolist())
                    a = agg.setdefault((thr, ma, kt, fold), [0.0, 0, 0, 0])
                    for n_gt, m in matches:
                        hits = {p: v for p, v in m.items() if p in kept_ids}
                        tp = len(hits)
                        a[0] += sum(v[1] for v in hits.values())
                        a[1] += tp
                        a[2] += len(kept_ids) - tp
                        a[3] += n_gt - len({v[0] for v in hits.values()})
        if n % 100 == 0:
            print("tune", n, len(stems), f"{time.time() - t0:.0f}s", flush=True)

    def pq(v):
        den = v[1] + 0.5 * v[2] + 0.5 * v[3]
        return v[0] / den if den else 0.0

    rows = []
    for thr in thrs:
        for ma in min_areas:
            for kt in keep_thrs:
                per = [agg[(thr, ma, kt, f)] for f in cfg["train"]["folds"]]
                tot = np.sum(np.array(per, dtype=object), axis=0)
                rows.append(dict(thr=thr, min_area=ma, keep_thr=kt, pq=pq(tot), per_fold=[pq(v) for v in per],
                                 tp=tot[1], fp=tot[2], fn=tot[3]))
    df = pd.DataFrame(rows).sort_values("pq", ascending=False)
    df.drop(columns="per_fold").to_csv(os.path.join(run_dir, "tune_grid.csv"), index=False)
    best = df.iloc[0]
    per_fold = [round(x, 5) for x in best.per_fold]
    out = {"metric": "panoptic_quality", "per_fold": per_fold, "mean": round(float(best.pq), 5),
           "std": round(float(np.std(per_fold)), 5), "post": {"thr": float(best.thr), "min_area": float(best.min_area),
                                                              "keep_thr": float(best.keep_thr)},
           "tp": int(best.tp), "fp": int(best.fp), "fn": int(best.fn), "eval_res": res, "n_train": len(stems)}
    json.dump(out, open(os.path.join(run_dir, "cv_score.json"), "w"), indent=1)
    print(json.dumps(out), flush=True)
    print(df.head(10).drop(columns="per_fold").to_string(), flush=True)
    return out


# ----------------------------------------------------------------------------- test inference
def rle_counts(mask):
    from pycocotools import mask as mask_utils
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return r["counts"].decode("ascii")


def predict(cfg, run_dir):
    import torch
    res = cfg["data"]["res"]
    post = json.load(open(os.path.join(run_dir, "cv_score.json")))["post"]
    test_dir = os.path.join(RAW, "test", "test_images")
    files = sorted(os.listdir(test_dir))
    models = []
    for fold in cfg["train"]["folds"]:
        m = build_model(dict(cfg, model=dict(cfg["model"], weights=None))).cuda()
        m.load_state_dict(torch.load(os.path.join(run_dir, f"model_f{fold}.pt"), map_location="cuda"))
        m.eval()
        models.append(m)
    os.makedirs(os.path.join(run_dir, "test_probs"), exist_ok=True)
    rows = []
    for fn in files:
        stem = os.path.splitext(fn)[0]
        img = imread(os.path.join(test_dir, fn), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        prob = np.mean([predict_tta(m, small, cfg["infer"]["tta"], "cuda") for m in models], axis=0)
        np.save(os.path.join(run_dir, "test_probs", stem + ".npy"), np.round(prob * 255).astype(np.uint8))
        prob_full = cv2.resize(prob, (FULL, FULL), interpolation=cv2.INTER_LINEAR)
        inst, _ = instances_from_prob(prob_full, post["thr"], post["min_area"], post["keep_thr"])
        for k in range(1, int(inst.max()) + 1):
            rows.append((f"{stem}_{k}", rle_counts(inst == k)))
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(os.path.join(run_dir, "test_preds.csv"), index=False)
    print("test rows", len(sub), "images with preds", sub.filament_id.str.rsplit("_", n=1).str[0].nunique(), "/", len(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "train", "tune", "predict", "all"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    run_dir = os.path.dirname(os.path.abspath(a.config))
    if os.name == "nt":  # keep the laptop from idle-sleeping while this process runs (ES_CONTINUOUS|ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if a.cmd == "cache":
        build_cache(cfg["data"]["res"])
    if a.cmd in ("train", "all"):
        train(cfg, run_dir)
    if a.cmd in ("tune", "all"):
        tune(cfg, run_dir)
    if a.cmd in ("predict", "all"):
        predict(cfg, run_dir)


if __name__ == "__main__":
    main()
