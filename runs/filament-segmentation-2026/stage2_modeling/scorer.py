"""Instance-level keep/drop scorer on top of a semantic run's OOF probabilities.

Host PQ arithmetic: keeping a predicted instance that matches k of the n annotation sets of its
image (IoUs s_j) changes numerator by sum(s_j) and denominator by exactly n/2, so it helps iff
    y := sum_j s_j / n  >  PQ / 2.
We regress y with LightGBM from per-instance features (fold-honest: trained on other folds' OOF)
and keep instances with y_hat > tau, tau tuned on the pooled OOF PQ.

Usage: python scorer.py --run runs/<run_id> [--thr 0.5]
Writes <run>/scorer/{features.parquet, model_f*.txt, scorer_cv.json}
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import FULL, PROC, gt_maps_for, imread, load_cache, make_folds  # noqa: E402

FEATS = ["area", "mean_p", "max_p", "std_p", "frac_hi", "bbox_h", "bbox_w", "fill", "elong", "r_disk",
         "int_in", "int_ring", "contrast", "n_inst_img", "img_mean_p", "rank_area"]
FEATS_V2 = ["frac_05", "frac_09", "n_sub_06", "perim_ratio", "edge_sharp"]


def feature_list(v2, n_members):
    feats = list(FEATS)
    if v2:
        feats += FEATS_V2
        feats += [f"mean_p_m{j}" for j in range(n_members)] + (["member_diff"] if n_members > 1 else [])
    return feats


def v2_features(prob, lab, n, member_probs):
    """Threshold-persistence, shape and ensemble-disagreement features (component-tree inspired)."""
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    frac_05 = ndimage.mean(prob > 0.5, lab, ids)
    frac_09 = ndimage.mean(prob > 0.9, lab, ids)
    # how many pieces each component breaks into at a stricter threshold (merge indicator)
    sub_lab, _ = ndimage.label((prob > 0.6) & (lab > 0), structure=np.ones((3, 3)))
    n_sub = np.zeros(n + 1)
    for parent, child in set(zip(lab[sub_lab > 0].tolist(), sub_lab[sub_lab > 0].tolist())):
        n_sub[parent] += 1
    area = ndimage.sum_labels(np.ones_like(prob), lab, ids)
    edge = (lab > 0) & (ndimage.grey_erosion(lab, size=(3, 3)) != lab)
    perim = ndimage.sum_labels(edge, lab, ids)
    gy, gx = np.gradient(prob)
    grad = np.hypot(gy, gx)
    edge_sharp = ndimage.mean(grad, np.where(edge, lab, 0), ids)
    rows = [dict(frac_05=frac_05[i], frac_09=frac_09[i], n_sub_06=n_sub[i + 1], perim_ratio=perim[i] / np.sqrt(area[i]),
                 edge_sharp=edge_sharp[i]) for i in range(n)]
    if member_probs:
        means = [ndimage.mean(mp, lab, ids) for mp in member_probs]
        for i, r in enumerate(rows):
            for j, m in enumerate(means):
                r[f"mean_p_m{j}"] = m[i]
            if len(means) > 1:
                r["member_diff"] = float(np.ptp([m[i] for m in means]))
    return rows


def component_features(prob, img, lab, n, scale):
    """Per-instance feature rows for label map lab (1..n) at working resolution."""
    if n == 0:
        return []
    h = prob.shape[0]
    ids = np.arange(1, n + 1)
    objs = ndimage.find_objects(lab)
    area = ndimage.sum_labels(np.ones_like(prob), lab, ids)
    mean_p = ndimage.mean(prob, lab, ids)
    max_p = ndimage.maximum(prob, lab, ids)
    std_p = ndimage.standard_deviation(prob, lab, ids)
    frac_hi = ndimage.mean(prob > 0.7, lab, ids)
    int_in = ndimage.mean(img, lab, ids)
    # ring of ~6px around each instance for local contrast
    dil = ndimage.grey_dilation(lab, size=(13, 13))
    ring = np.where((lab == 0), dil, 0)
    int_ring = ndimage.mean(img, ring, ids)
    disk = img > 30
    yy, xx = np.nonzero(disk)
    cy, cx, rad = yy.mean(), xx.mean(), np.sqrt(disk.sum() / np.pi)
    com = ndimage.center_of_mass(np.ones_like(prob), lab, ids)
    rows = []
    for i, sl in enumerate(objs):
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        sub = lab[sl] == i + 1
        ys, xs = np.nonzero(sub)
        if len(ys) > 2:
            ev = np.linalg.eigvalsh(np.cov(np.stack([ys, xs]).astype(np.float64)) + 1e-6 * np.eye(2))
            elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        rows.append(dict(area=area[i] * scale, mean_p=mean_p[i], max_p=max_p[i], std_p=std_p[i], frac_hi=frac_hi[i],
                         bbox_h=bh * np.sqrt(scale), bbox_w=bw * np.sqrt(scale), fill=area[i] / (bh * bw), elong=elong,
                         r_disk=np.hypot(com[i][0] - cy, com[i][1] - cx) / rad, int_in=int_in[i], int_ring=int_ring[i],
                         contrast=int_ring[i] - int_in[i]))
    order = np.argsort(-area)
    rank = np.empty(n)
    rank[order] = np.arange(n)
    for i, r in enumerate(rows):
        r.update(n_inst_img=n, img_mean_p=float(prob[disk].mean()), rank_area=rank[i])
    return rows


def build_features(run, res, thr, v2=False, members=()):
    idx = pd.read_csv(os.path.join(PROC, "index.csv"))
    folds = make_folds()
    imgs, _, stems = load_cache(res)
    img_of = dict(zip(stems, range(len(stems))))
    scale = (FULL / res) ** 2
    out = []
    for stem, fold in zip(folds.stem, folds.fold):
        p = os.path.join(run, "oof", stem + ".npy")
        if not os.path.exists(p):
            continue
        prob = np.load(p).astype(np.float32) / 255.0
        img = imgs[img_of[stem]].astype(np.float32)
        lab, n = ndimage.label(prob > thr, structure=np.ones((3, 3)))
        rows = component_features(prob, img, lab, n, scale)
        if v2:
            mps = [np.load(os.path.join(m, "oof", stem + ".npy")).astype(np.float32) / 255.0 for m in members]
            for r, e in zip(rows, v2_features(prob, lab, n, mps)):
                r.update(e)
        gts = gt_maps_for(stem, res, idx)
        ious = np.zeros((n + 1, len(gts)))
        ngt = []
        for j, g in enumerate(gts):
            g = g.astype(np.int64)
            ngt.append(len(np.unique(g)) - 1)
            both = (g > 0) & (lab > 0)
            base = n + 1
            keys, inter = np.unique(g[both] * base + lab[both], return_counts=True)
            ga = np.bincount(g.ravel(), minlength=int(g.max()) + 1)
            pa = np.bincount(lab.ravel(), minlength=base)
            for k, it in zip(keys.tolist(), inter.tolist()):
                gi, pi = divmod(k, base)
                iou = it / (ga[gi] + pa[pi] - it)
                if iou > 0.5:
                    ious[pi, j] = iou
        for i, r in enumerate(rows):
            r.update(stem=stem, fold=fold, inst=i + 1, n_sets=len(gts), y=ious[i + 1].mean(),
                     **{f"iou_{j}": ious[i + 1, j] for j in range(3) if j < len(gts)})
            out.append(r)
        # image-level GT counts (for FN accounting) stored on a sentinel row
        out.append(dict(stem=stem, fold=fold, inst=0, n_sets=len(gts), n_gt_total=sum(ngt)))
    return pd.DataFrame(out)


def pooled_pq(df, keep):
    """df rows = instances (inst>0) + sentinel rows (inst==0) with n_gt_total; keep = bool mask over instances."""
    inst = df[df.inst > 0]
    iou_cols = [c for c in inst.columns if c.startswith("iou_")]
    m = inst[iou_cols].fillna(0).to_numpy()
    k = keep.astype(bool)
    sum_iou = m[k].sum()
    tp = (m[k] > 0).sum()
    n_pred_sets = inst.n_sets.to_numpy()[k].sum()       # each kept pred is scored once per annotation set
    fp = n_pred_sets - tp
    fn = df.loc[df.inst == 0, "n_gt_total"].sum() - tp
    den = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / den if den else 0.0


def predict_test(run, res, thr, tau, out_csv, scorer_dir="scorer", v2=False, members=()):
    """Apply fold-averaged scorer to <run>/test_probs/*.npy and write a submission CSV.
    Selection happens at working resolution; final masks are re-thresholded on the bilinearly
    upsampled prob at 2048 and kept if they overlap a selected component by > 50%."""
    import lightgbm as lgb
    from pipeline import RAW, rle_counts
    models = [lgb.Booster(model_file=os.path.join(run, scorer_dir, f))
              for f in sorted(os.listdir(os.path.join(run, scorer_dir))) if f.startswith("model_f")]
    feats_used = feature_list(v2, len(members))
    test_dir = os.path.join(RAW, "test", "test_images")
    scale = (FULL / res) ** 2
    rows, n_img_with = [], 0
    for fn in sorted(os.listdir(test_dir)):
        stem = os.path.splitext(fn)[0]
        prob = np.load(os.path.join(run, "test_probs", stem + ".npy")).astype(np.float32) / 255.0
        img = cv2.resize(imread(os.path.join(test_dir, fn), cv2.IMREAD_GRAYSCALE), (res, res),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
        lab, n = ndimage.label(prob > thr, structure=np.ones((3, 3)))
        if n == 0:
            continue
        rows_f = component_features(prob, img, lab, n, scale)
        if v2:
            mps = [np.load(os.path.join(m, "test_probs", stem + ".npy")).astype(np.float32) / 255.0 for m in members]
            for r, e in zip(rows_f, v2_features(prob, lab, n, mps)):
                r.update(e)
        feats = pd.DataFrame(rows_f)
        y_hat = np.mean([m.predict(feats[feats_used]) for m in models], axis=0)
        keep_ids = np.where(y_hat > tau)[0] + 1
        sel = np.isin(lab, keep_ids).astype(np.uint8)
        sel_full = cv2.resize(sel, (FULL, FULL), interpolation=cv2.INTER_NEAREST)
        prob_full = cv2.resize(prob, (FULL, FULL), interpolation=cv2.INTER_LINEAR)
        lab_full, n_full = ndimage.label(prob_full > thr, structure=np.ones((3, 3)))
        if n_full == 0:
            continue
        frac = ndimage.mean(sel_full, lab_full, np.arange(1, n_full + 1))
        k = 0
        for i in np.where(frac > 0.5)[0] + 1:
            k += 1
            rows.append((f"{stem}_{k}", rle_counts(lab_full == i)))
        n_img_with += k > 0
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(out_csv, index=False)
    print("test rows", len(sub), "images with preds", n_img_with)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--predict", action="store_true", help="write <run>/test_preds_scorer.csv using scorer_cv tau")
    ap.add_argument("--v2", action="store_true", help="extra persistence/shape features (+ member features)")
    ap.add_argument("--members", nargs="*", default=[], help="member run dirs (ensemble disagreement features)")
    a = ap.parse_args()
    sdir = "scorer_v2" if a.v2 else "scorer"
    feats_used = feature_list(a.v2, len(a.members))
    if a.predict:
        cv = json.load(open(os.path.join(a.run, sdir, f"scorer_cv_t{a.thr}.json")))
        name = "test_preds_scorer_v2.csv" if a.v2 else "test_preds_scorer.csv"
        predict_test(a.run, a.res, a.thr, cv["best_tau"], os.path.join(a.run, name), sdir, a.v2, a.members)
        return
    out_dir = os.path.join(a.run, sdir)
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"features_t{a.thr}.pkl")
    if os.path.exists(fp):
        df = pd.read_pickle(fp)
    else:
        df = build_features(a.run, a.res, a.thr, a.v2, a.members)
        df.to_pickle(fp)
    inst = df[df.inst > 0].copy()
    inst["y_hat"] = np.nan
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    for f in sorted(inst.fold.unique()):
        tr, va = inst.fold != f, inst.fold == f
        if tr.sum() == 0:
            continue
        m = lgb.train(params, lgb.Dataset(inst.loc[tr, feats_used], inst.loc[tr, "y"]), num_boost_round=400)
        inst.loc[va, "y_hat"] = m.predict(inst.loc[va, feats_used])
        m.save_model(os.path.join(out_dir, f"model_f{f}.txt"))
    inst = inst[inst.y_hat.notna()]
    sub = pd.concat([inst, df[(df.inst == 0) & df.stem.isin(inst.stem)]])
    sub_inst = sub[sub.inst > 0]
    base = pooled_pq(sub, (sub_inst.area >= 500).to_numpy())
    res = []
    for tau in np.arange(0.05, 0.45, 0.01):
        keep = (sub_inst.y_hat > tau).to_numpy()
        per = {int(f): pooled_pq(sub[sub.fold == f], keep[(sub_inst.fold == f).to_numpy()]) for f in sorted(sub_inst.fold.unique())}
        res.append((round(tau, 2), pooled_pq(sub, keep), per))
    best = max(res, key=lambda r: r[1])
    out = dict(thr=a.thr, baseline_min_area500=round(base, 5), best_tau=best[0], pq=round(best[1], 5),
               per_fold={k: round(v, 5) for k, v in best[2].items()}, n_instances=len(sub_inst),
               tau_curve=[(float(t), round(float(p), 5)) for t, p, _ in res])
    json.dump(out, open(os.path.join(out_dir, f"scorer_cv_t{a.thr}.json"), "w"), indent=1)
    print(json.dumps(out))
    print("tau curve:", [(t, round(p, 4)) for t, p, _ in res[::3]])


if __name__ == "__main__":
    main()
