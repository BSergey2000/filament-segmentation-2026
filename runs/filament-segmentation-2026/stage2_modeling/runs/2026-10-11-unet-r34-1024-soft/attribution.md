# Attribution — 2026-10-11-unet-r34-1024-soft

Own re-implementation in stage2_modeling/pipeline.py (no code copied).

## Ideas used (from stage1_recon/ideas_pool.md)
- cv: grouped-by-observation split — dattadhebe/solar-filament-grouped-cv-and-strategy-writeup, tushhaaarrrr/solar-filament-segmentation-2026-0-40
- cv: exact host PQ evaluator (pooled over annotator sets, IoU>0.5) — azimahmadzadeh/self-evaluation-notebook
- feature: soft annotator-mean target — tushhaaarrrr/solar-filament-segmentation-2026-0-40, dattadhebe/solar-filament-grouped-cv-and-strategy-writeup
- model: semantic U-Net + connected components — lamhuy8904/solar-filament-unet-segmentation-0-55, anthonytherrien/magfilo-blended-tile-u-net-for-solar-filaments
- loss: BCE + soft Dice — tushhaaarrrr/solar-filament-segmentation-2026-0-40
- post: metric-aware instance inclusion (mean-prob keep threshold) + min area — tushhaaarrrr/solar-filament-segmentation-2026-0-40, juanmirod/solar-filament-2026-final-report
- aug: dihedral TTA (4 views) — phuongncn/lb-0-70-topology-safe-solar-filaments-handoff

## Own additions
- 1024 downsample with 512 random crops (4GB VRAM), 1-channel input, gamma/brightness jitter.
- Analytic post-processing grid: per-instance match records reused across min_area/keep_thr (instances disjoint).

## Result
- CV PQ (5-fold, pooled, eval@1024): 0.38639 ± 0.00719 with thr 0.35, min_area 500, keep_thr 0.7. Fold PQ at 2048 ≈ +0.003 vs 1024.
- + LightGBM instance scorer (scorer.py, thr 0.4, tau 0.20): 0.40081 (+0.0144), every fold improved.
- Ideas that earned their keep: metric-aware inclusion (+0.014 via scorer), soft target + BCE/Dice baseline.
- Scorer variant citations (post: component-tree + hurdle scorer + global tau): tushhaaarrrr/solar-filament-segmentation-2026-0-40, juanmirod/solar-filament-2026-final-report
