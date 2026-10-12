"""Sanity-check a submission CSV before upload.

Checks: header, unique ids, stem in test set, RLE decodes to 2048x2048, no empty masks,
no overlapping masks within an image, orientation (mask overlaps dark pixels of the image more
than its transpose does, aggregated).
Usage: python validate_submission.py <csv> [--probs-dir <run>/test_probs]
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "data", "raw", "MAGFiLO_1.0_Kaggle_2026", "test", "test_images")


def main(path):
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["filament_id", "segmentation_rle"], sub.columns
    assert sub.filament_id.is_unique, "duplicate filament_id"
    stems = {os.path.splitext(f)[0] for f in os.listdir(TEST)}
    sub["stem"] = sub.filament_id.str.rsplit("_", n=1).str[0]
    bad = set(sub.stem) - stems
    assert not bad, f"unknown stems: {list(bad)[:5]}"
    assert not sub.segmentation_rle.str.contains('"|\'').any(), "quotes in RLE"
    darker, darker_t, n_empty, n_overlap = 0.0, 0.0, 0, 0
    for stem, g in sub.groupby("stem"):
        img = cv2.imdecode(np.fromfile(os.path.join(TEST, stem + ".jpeg"), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        cover = np.zeros((2048, 2048), np.uint8)
        for c in g.segmentation_rle:
            m = mask_utils.decode({"size": [2048, 2048], "counts": c.encode()})
            if m.sum() == 0:
                n_empty += 1
                continue
            n_overlap += int((cover & m).any())
            cover |= m
        if cover.any():
            darker += img[cover > 0].mean() - img[img > 30].mean()
            darker_t += img[cover.T > 0].mean() - img[img > 30].mean()
    print(f"rows={len(sub)} images={sub.stem.nunique()}/{len(stems)} empty={n_empty} overlapping_masks={n_overlap}")
    print(f"orientation: mask-darkness {darker:.1f} vs transposed {darker_t:.1f} (mask should be darker = more negative)")
    ok = n_empty == 0 and n_overlap == 0 and darker < darker_t
    print("VALID" if ok else "INVALID")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
