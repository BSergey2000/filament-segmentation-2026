# Modeling notes

## 2026-10-10 — baseline fold-0 diagnostics (unet-r34-1024-soft, thr 0.5, min_area 500)
- Fold-0 OOF PQ 0.397 (eval at 1024). Post-proc plateau: min_area 500 best, keep_thr (mean prob) adds only +0.001.
- FN (766 over annotator sets): 49% fully missed (median area ~980 px full-res — small, low-agreement),
  16% IoU 0–0.3, 34% near-miss IoU 0.3–0.5. Split-into-several-preds: 55.
- FP (767): 34% no overlap, 31% IoU 0–0.3, 35% near-miss 0.3–0.5. Merged multi-GT: 22.
- => Main lever is mask shape/boundary quality (near-misses convert FP+FN → TP). Next: native 2048 res,
  then instance-level refinement. Split/merge handling is low priority.

## 2026-10-10 — instance scorer (scorer.py) on folds 0-1
- LightGBM regresses y = mean over annotator sets of matched IoU (0 if ≤0.5); keep iff y_hat > tau.
- Baseline (thr 0.5, min_area 500) 0.3892 → scorer 0.4022 (+0.013), per fold 0.4073 / 0.3958. Best tau 0.20 ≈ PQ/2 (theory holds).
- Strongest features: mean_p, frac_hi, area/bbox, rank_area, fill. Trained on only 1 other fold — expect more gain with 4.

## 2026-10-11 — native 2048 ablation (fold 0): NEGATIVE
- unet-r34-2048-soft, crop 640, batch 4: fold-0 PQ 0.326 vs 0.397 at 1024. Context per crop matters more than pixel resolution. Killed after fold 0.
- Public LB: scorer variant 0.34, cc-postproc 0.33 (CV 0.401 / 0.386) → gap −0.06, rank-preserving.
- Next: fold-0 ablations at 1024: crop 768 (more context), other encoders, longer training.

## 2026-10-11 — crop-size ablation (fold 0, 1024 res): BIG WIN
- crop 768 / batch 4: fold-0 PQ 0.4291 vs 0.3979 ref (+0.031); final loss 0.178 vs 0.20. 113 s/epoch.
- Confirms: context per crop is the main lever. Queued crop 1024 (full image) / batch 3.
- effb3 encoder (crop 512): fold-0 PQ 0.4188 vs 0.3979 (+0.021). Orthogonal to crop → combine effb3 + larger crop next.
- ep50 (crop 512, r34): fold-0 PQ 0.4026 (+0.005) — within noise, not worth 1.7x time. Keep 30 epochs.
- crop 1024 (full image, batch 3): fold-0 PQ 0.4266 ≈ crop768 0.4291 → context gain saturates at 768; 768 is 2x faster (113 vs 200 s/epoch). Chosen: crop 768.

## 2026-10-11 — crop768 full run + submission
- 2026-10-11-unet-r34-1024-crop768: CV 0.4086 ± 0.008 (+0.022 vs crop512); + scorer (thr 0.3): 0.4239 ± 0.0076. Public LB 0.35 (prev best 0.34).
- OOF prob-average ensemble 0.4·crop512 + 0.6·crop768 → scorer 0.4176 < 0.4239: weaker member hurts. Deleted.
- Running: abl-f0-effb3-crop768 (fold 0). Target to beat on fold 0: 0.4291 (r34 crop768).
- effb3 + crop768 (batch 2; batch 4 spilled VRAM → 1500 s/epoch): fold-0 0.4335 (+0.004 vs r34-768). Fold-0 50/50 prob ensemble r34-768 + effb3-768: 0.4405 (+0.011). → training effb3-768 full 5-fold as ensemble member.

## 2026-10-12 — effb3-crop768 full run + ensemble
- effb3-crop768: CV 0.4135 ± 0.009; + scorer 0.4239 (same as r34-768 + scorer).
- 50/50 ensemble r34-768 + effb3-768 + scorer (thr 0.3, tau 0.24): CV 0.4293; folds 0.451/0.423/0.419/0.432/0.417 (4/5 up vs r34-768). Public LB 0.35 (unchanged at 2-decimal resolution).
- Laptop slept ~5h during fold 4 (lid closed?) despite SetThreadExecutionState.
- Public LB too coarse (~90 imgs, 2 decimals) to resolve <0.01 gains → rely on CV.
- scorer v2 (+frac_05/09, n_sub_06, perim_ratio, edge_sharp, per-member mean prob, member_diff) on ens r34+effb3: 0.4315 vs v1 0.4293 (+0.002, 4/5 folds up). test_preds_scorer_v2.csv built (not submitted: below noise / LB resolution).
- UNet++ r34 crop768 batch 2: 249 s/epoch (2.2x Unet) → fold-0 ablation ~2h; full run ~10h, decide after fold 0.
- UNet++ r34 crop768 (fold 0): 0.4303 single; 3-way ens (r34-768, effb3-768, unet++) fold-0 0.4413 vs 2-way 0.4405 (+0.001) → not worth ~8h full run. Abandoned.
- TTA8 re-inference (no retraining): r34-768 0.4086→0.4126, effb3-768 0.4135→0.4173; ensemble+scorer v2 0.43147→0.43165 (gain absorbed). Plateau ≈ 0.43 CV.
- Next: resolution at equal context: res 1536, crop 1152 (= 768@1024 field of view), batch 2 (fold 0).
- res 1536 / crop 1152 (equal context to 768@1024), fold 0: 0.4285 vs 0.4291 → resolution gives nothing at equal context. As 3rd ensemble member (resized to 1024): fold-0 0.4444 vs 0.4441. Abandoned.
- CONCLUSION (2026-10-12): model-side levers saturated (resolution, 3rd architecture, epochs, TTA8 all < +0.005 after ensembling). Limit is annotator disagreement (human-vs-human PQ 0.34). Best = ens r34+effb3 crop768 TTA8 + scorer v2, CV 0.4317.
