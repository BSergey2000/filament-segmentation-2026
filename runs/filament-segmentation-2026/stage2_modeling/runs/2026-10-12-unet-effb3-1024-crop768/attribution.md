# Attribution — 2026-10-12-unet-effb3-1024-crop768

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
- EfficientNet-B3 encoder, 1024 input, 768 crops, batch 2 (4GB VRAM); fold 0 reused from abl-f0-effb3-crop768. Intended as ensemble member with r34-crop768 (fold-0 ensemble 0.4405 vs 0.4291 single).
- Analytic post-processing grid: per-instance match records reused across min_area/keep_thr (instances disjoint).

## Result
- CV PQ 0.41346 ± 0.00893; + scorer (thr 0.3): 0.42387. As ensemble member with r34-crop768 (50/50 probs) + scorer: 0.42934, public LB 0.35.
- Scorer citations: tushhaaarrrr/solar-filament-segmentation-2026-0-40, juanmirod/solar-filament-2026-final-report
