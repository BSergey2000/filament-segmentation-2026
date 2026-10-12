"""Build a cache of training data from the competition COCO json.

Outputs (under data/processed/):
  images/<stem>.png          grayscale uint8, full 2048x2048 (lossless copy of jpeg decode)
  inst/<annot_set_id>.png    uint16 instance label map, 0 = background, 1..K = filaments
  index.csv                  annot_set_id, stem, n_filaments, annotator
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(RUN, "data", "raw", "MAGFiLO_1.0_Kaggle_2026")
OUT = os.path.join(RUN, "data", "processed")
SIZE = 2048


def imread(path, flags=cv2.IMREAD_UNCHANGED):
    # cv2.imread cannot open non-ASCII Windows paths (project lives under "Документы")
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    assert ok, path
    buf.tofile(path)


def polygon_to_mask(seg):
    rles = mask_utils.frPyObjects(seg, SIZE, SIZE)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "inst"), exist_ok=True)
    coco = json.load(open(os.path.join(RAW, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), encoding="utf-8"))
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    rows = []
    for k, im in enumerate(coco["images"]):
        stem = os.path.splitext(im["file_name"])[0]
        img_out = os.path.join(OUT, "images", stem + ".png")
        if not os.path.exists(img_out):
            img = imread(os.path.join(RAW, "train", "train_images", im["file_name"]), cv2.IMREAD_GRAYSCALE)
            assert img.shape == (SIZE, SIZE), img.shape
            imwrite(img_out, img)
        anns = sorted(anns_by_img.get(im["id"], []), key=lambda a: -a["area"])
        inst = np.zeros((SIZE, SIZE), np.uint16)
        # paint large first so small filaments drawn on top stay visible where they overlap
        for i, a in enumerate(anns, 1):
            inst[polygon_to_mask(a["segmentation"])] = i
        imwrite(os.path.join(OUT, "inst", im["id"] + ".png"), inst)
        rows.append(dict(annot_set_id=im["id"], stem=stem, n_filaments=len(anns), annotator=im["id"].split("-")[0]))
        if k % 100 == 0:
            print(k, len(coco["images"]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "index.csv"), index=False)
    print("done", len(rows))


if __name__ == "__main__":
    sys.exit(main())
