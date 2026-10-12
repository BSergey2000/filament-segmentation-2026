# Solar filament instance segmentation — MAGFiLO / IEEE BigData Cup 2026

Instance segmentation of solar filaments in 2048×2048 GONG H-alpha images, scored by the host's
Panoptic Quality (PQ). Everything runs on a single 4 GB laptop GPU (RTX 3050).

## Result

| Model | 5-fold CV PQ | Public LB |
|---|---|---|
| U-Net ResNet34 + U-Net EfficientNet-B3 (crop 768) ensemble + LightGBM instance scorer v2 | **0.432** | 0.35 (v1) |
| U-Net ResNet34, crop 768 + scorer | 0.424 | 0.35 |
| U-Net ResNet34, crop 512 + scorer | 0.401 | 0.34 |
| U-Net ResNet34, crop 512, connected components only | 0.386 | 0.33 |

For reference, two independent human annotators agree at PQ 0.340 under the same metric.

## Pipeline

1. **Data** (`prep_data.py`): COCO polygons → one instance label map per annotation set; images kept as
   grayscale. Each image has 1–3 independent annotation sets.
2. **CV**: GroupKFold(5) by physical image, so annotation sets of one image never straddle folds.
3. **Target**: soft annotator-mean mask (fraction of annotators marking each pixel).
4. **Model** (`pipeline.py`): U-Net (segmentation_models_pytorch), 1-channel input, image resized to
   1024, trained on random 768 crops (context matters more than resolution: native 2048 with 640
   crops lost 0.07 PQ), BCE + 0.5·soft-Dice, AMP, 30 epochs, OneCycle. Dihedral flips/rotations and
   brightness/contrast/gamma jitter.
5. **Inference**: 8-view dihedral TTA, 5-fold average, 50/50 average of ResNet34 and EfficientNet-B3 models.
6. **Instances**: threshold 0.3 → 8-connected components (no morphology: it hurt).
7. **Instance scorer** (`scorer.py`): the host PQ adds Σ IoU to the numerator and exactly n/2 to the
   denominator when a predicted filament is kept against n annotation sets, so a filament should be
   kept iff its mean matched IoU across annotators exceeds PQ/2. A LightGBM regressor predicts that
   quantity from per-component features (probability statistics, threshold persistence, shape,
   local contrast, disk position, member disagreement); components with prediction > τ (≈0.23,
   ≈ PQ/2 as predicted by the derivation) are kept. +0.014–0.02 PQ.
8. **Submission**: kept components re-thresholded on the bilinearly upsampled 2048 probability map,
   encoded as COCO RLE (Fortran order), validated by `validate_submission.py`.

`pq.py` is an exact re-implementation of the host's self-evaluation metric (pooled over annotation
sets, every pair with IoU > 0.5 counts as TP); it reproduces the inter-annotator PQ of 0.340.

## Reproduce

```bash
pip install -r requirements.txt
cd runs/filament-segmentation-2026/stage2_modeling
python prep_data.py
python pipeline.py all --config runs/2026-10-11-unet-r34-1024-crop768/config.yaml
python pipeline.py all --config runs/2026-10-12-unet-effb3-1024-crop768/config.yaml
python reinfer.py --src 2026-10-11-unet-r34-1024-crop768 --dst 2026-10-12-unet-r34-1024-crop768-tta8 --tta 8
python reinfer.py --src 2026-10-12-unet-effb3-1024-crop768 --dst 2026-10-12-unet-effb3-1024-crop768-tta8 --tta 8
# then pipeline.py tune/predict on both -tta8 runs, ensemble.py, scorer.py --v2 [--predict]
```

Competition data goes in `runs/filament-segmentation-2026/data/raw/` (`kaggle competitions download -c filament-segmentation-2026`).
The public MAGFiLO release is deliberately **not** used: it contains the test labels.

## Credits

Ideas re-implemented from public Kaggle kernels (no code copied). Each run's `attribution.md` lists
the exact sources; main ones: tushhaaarrrr/solar-filament-segmentation-2026-0-40 (soft target,
instance keep threshold), azimahmadzadeh/self-evaluation-notebook (metric),
dattadhebe/solar-filament-grouped-cv-and-strategy-writeup (grouped CV),
juanmirod/solar-filament-2026-final-report (operating-point selection).
