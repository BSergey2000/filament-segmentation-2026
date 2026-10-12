"""Local replica of the host's Panoptic Quality (azimahmadzadeh/self-evaluation-notebook).

Host semantics:
  - loop over every GT annotation set (annotator x image); compare against all preds of that image
  - every (gt, pred) pair with IoU > 0.5 (strict) is a TP and contributes its IoU
  - FP = preds with no hit, FN = gts with no hit
  - PQ = sum(TP IoU) / (TP + 0.5 FP + 0.5 FN), pooled (micro) over all annotation sets
Works on instance label maps (0 = background) — equivalent to RLE when instances are disjoint.
"""
import numpy as np


def pq_counts(gt_inst, pred_inst):
    """Return (sum_iou, tp, fp, fn) for one annotation set vs. one prediction map."""
    gt_inst = gt_inst.astype(np.int64).ravel()
    pred_inst = pred_inst.astype(np.int64).ravel()
    gt_ids, gt_area = np.unique(gt_inst, return_counts=True)
    pr_ids, pr_area = np.unique(pred_inst, return_counts=True)
    gt_area = dict(zip(gt_ids.tolist(), gt_area.tolist()))
    pr_area = dict(zip(pr_ids.tolist(), pr_area.tolist()))
    gt_area.pop(0, None)
    pr_area.pop(0, None)

    both = (gt_inst > 0) & (pred_inst > 0)
    pairs = gt_inst[both] * (int(pred_inst.max()) + 1) + pred_inst[both]
    keys, inter = np.unique(pairs, return_counts=True)

    sum_iou, tp = 0.0, 0
    gt_hit, pr_hit = set(), set()
    base = int(pred_inst.max()) + 1
    for key, it in zip(keys.tolist(), inter.tolist()):
        g, p = divmod(key, base)
        iou = it / (gt_area[g] + pr_area[p] - it)
        if iou > 0.5:
            sum_iou += iou
            tp += 1
            gt_hit.add(g)
            pr_hit.add(p)
    fp = len(pr_area) - len(pr_hit)
    fn = len(gt_area) - len(gt_hit)
    return sum_iou, tp, fp, fn


class PQAccumulator:
    def __init__(self):
        self.sum_iou = 0.0
        self.tp = self.fp = self.fn = 0

    def add(self, gt_inst, pred_inst):
        s, tp, fp, fn = pq_counts(gt_inst, pred_inst)
        self.sum_iou += s
        self.tp += tp
        self.fp += fp
        self.fn += fn

    @property
    def score(self):
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.sum_iou / den if den else 0.0

    def summary(self):
        sq = self.sum_iou / self.tp if self.tp else 0.0
        return dict(pq=round(self.score, 5), sq=round(sq, 4), tp=self.tp, fp=self.fp, fn=self.fn)


if __name__ == "__main__":
    a = np.zeros((10, 10), np.int32); a[:5, :5] = 1; a[6:, 6:] = 2
    b = np.zeros((10, 10), np.int32); b[:5, :4] = 7; b[0, 9] = 3
    acc = PQAccumulator(); acc.add(a, b)
    # gt1 vs pred7: iou 20/25 = 0.8 -> TP; gt2 FN; pred3 FP -> 0.8 / (1 + 0.5 + 0.5) = 0.4
    assert abs(acc.score - 0.4) < 1e-9, acc.summary()
    print("ok", acc.summary())
