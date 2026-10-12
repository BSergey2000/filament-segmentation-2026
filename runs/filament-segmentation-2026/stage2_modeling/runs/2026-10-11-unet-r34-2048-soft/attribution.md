# Attribution — 2026-10-11-unet-r34-2048-soft

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
- Native 2048 resolution, 640 random crops, batch 4 (single change vs 1024 baseline), 1-channel input, gamma/brightness jitter.
- Analytic post-processing grid: per-instance match records reused across min_area/keep_thr (instances disjoint).

## Result
(filled after run)

## Result
- ABANDONED after fold 0: fold-0 PQ 0.3257 vs 0.3970 for the 1024 baseline (same post-proc grid). 640 crops at 2048 = 4x less context than 512@1024; final loss 0.232 vs 0.20.
