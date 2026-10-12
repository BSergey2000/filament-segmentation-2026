# Attribution — 2026-10-11-unet-r34-1024-crop768

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
- 1024 input with 768 random crops, batch 4 (fold-0 ablation +0.031 vs 512); fold 0 reused from abl-f0-crop768 (identical config+seed), 1-channel input, gamma/brightness jitter.
- Analytic post-processing grid: per-instance match records reused across min_area/keep_thr (instances disjoint).

## Result
- CV PQ 0.40861 ± 0.00795 (thr 0.3, min_area 300, keep 0.7); + scorer (thr 0.3, tau 0.21): 0.42385 ± 0.0076. Public LB 0.35.
- Scorer citations: tushhaaarrrr/solar-filament-segmentation-2026-0-40, juanmirod/solar-filament-2026-final-report
