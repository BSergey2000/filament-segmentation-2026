"""Generate pipeline.ipynb (repo root): the full pipeline, end to end, with skip-if-done steps."""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Solar filament instance segmentation — full pipeline\n\n"
       "MAGFiLO / IEEE BigData Cup 2026. Reproduces the final submission "
       "(U-Net ResNet34 + U-Net EfficientNet-B3 ensemble, 8-view TTA, LightGBM instance scorer; "
       "grouped 5-fold CV PQ 0.432).\n\n"
       "Every step is skipped when its outputs already exist; set `RETRAIN = True` to recompute "
       "everything from the raw competition data (~12 h on a 4 GB RTX 3050)."),
    code("import os, sys, json, subprocess\n"
         "import numpy as np, pandas as pd\n\n"
         "RETRAIN = False\n"
         "ROOT = os.path.abspath('.')\n"
         "S2 = os.path.join(ROOT, 'runs', 'filament-segmentation-2026', 'stage2_modeling')\n"
         "sys.path.insert(0, S2)\n\n"
         "def run(*args):\n"
         "    \"\"\"Run a pipeline script from the modeling dir and stream its last lines.\"\"\"\n"
         "    out = subprocess.run([sys.executable, *args], cwd=S2, capture_output=True, text=True,\n"
         "                         env={**os.environ, 'PYTHONWARNINGS': 'ignore'})\n"
         "    print('\\n'.join((out.stdout + out.stderr).strip().splitlines()[-6:]))\n"
         "    out.check_returncode()\n\n"
         "def done(path):\n"
         "    return os.path.exists(os.path.join(S2, path)) and not RETRAIN"),
    md("## 1. Data\nDownload with `kaggle competitions download -c filament-segmentation-2026` into "
       "`runs/filament-segmentation-2026/data/raw/` and unzip. `prep_data.py` turns the COCO polygons into "
       "one instance label map per annotation set (images have 1–3 independent annotators)."),
    code("if not done('../data/processed/index.csv'):\n"
         "    run('prep_data.py')\n"
         "idx = pd.read_csv(os.path.join(S2, '..', 'data', 'processed', 'index.csv'))\n"
         "print(len(idx), 'annotation sets on', idx.stem.nunique(), 'images')\n"
         "print('annotation sets per image:', idx.groupby('stem').size().value_counts().sort_index().to_dict())\n"
         "print('filaments per set: median', idx.n_filaments.median(), 'max', idx.n_filaments.max())"),
    md("## 2. Metric and validation\n`pq.py` re-implements the host PQ exactly: loop over annotation sets, every "
       "pair with IoU > 0.5 is a TP, counts pooled over all sets. Scoring each annotator against the others gives "
       "the human agreement level. CV is GroupKFold(5) by physical image (`folds.csv`)."),
    code("from pipeline import make_folds, gt_maps_for\n"
         "from pq import PQAccumulator\n"
         "acc = PQAccumulator()\n"
         "for stem, g in idx.groupby('stem'):\n"
         "    if len(g) < 2: continue\n"
         "    maps = gt_maps_for(stem, 1024, idx)\n"
         "    for i in range(len(maps)):\n"
         "        for j in range(len(maps)):\n"
         "            if i != j: acc.add(maps[j], maps[i])\n"
         "print('human vs human PQ (1024 px):', acc.summary())\n"
         "print(make_folds().fold.value_counts().sort_index().to_dict())"),
    md("## 3. Train the two ensemble members\nU-Net, 1-channel input resized to 1024, random 768 crops, soft "
       "annotator-mean target, BCE + 0.5 soft-Dice, 30 epochs. `pipeline.py all` = train (fold-resumable) → "
       "OOF → post-processing grid → test inference."),
    code("MEMBERS = ['2026-10-11-unet-r34-1024-crop768', '2026-10-12-unet-effb3-1024-crop768']\n"
         "for r in MEMBERS:\n"
         "    if not done(f'runs/{r}/cv_score.json'):\n"
         "        run('pipeline.py', 'all', '--config', f'runs/{r}/config.yaml')\n"
         "    print(r, json.load(open(os.path.join(S2, 'runs', r, 'cv_score.json')))['mean'])"),
    md("## 4. 8-view TTA re-inference\nSame weights, OOF and test probabilities recomputed with all 8 dihedral views."),
    code("TTA8 = [r.replace('2026-10-11-unet', '2026-10-12-unet') + '-tta8' for r in MEMBERS]\n"
         "for src, dst in zip(MEMBERS, TTA8):\n"
         "    if not done(f'runs/{dst}/test_preds.csv'):\n"
         "        run('reinfer.py', '--src', src, '--dst', dst, '--tta', '8')\n"
         "        run('pipeline.py', 'tune', '--config', f'runs/{dst}/config.yaml')\n"
         "        run('pipeline.py', 'predict', '--config', f'runs/{dst}/config.yaml')\n"
         "    print(dst, json.load(open(os.path.join(S2, 'runs', dst, 'cv_score.json')))['mean'])"),
    md("## 5. Ensemble + instance scorer\nProbabilities averaged 50/50. Components at threshold 0.3 are scored by "
       "LightGBM, which predicts the mean matched IoU across annotators; the host PQ adds exactly n/2 to its "
       "denominator per kept instance, so a component is worth keeping iff that value exceeds PQ/2."),
    code("ENS = '2026-10-12-ens-r34-effb3-768-tta8'\n"
         "members = [f'runs/{r}' for r in TTA8]\n"
         "if not done(f'runs/{ENS}/test_probs'):\n"
         "    run('ensemble.py', '--out', ENS, '--runs', *TTA8)\n"
         "if not done(f'runs/{ENS}/scorer_v2/scorer_cv_t0.3.json'):\n"
         "    run('scorer.py', '--run', f'runs/{ENS}', '--thr', '0.3', '--v2', '--members', *members)\n"
         "cv = json.load(open(os.path.join(S2, 'runs', ENS, 'scorer_v2', 'scorer_cv_t0.3.json')))\n"
         "print('final CV PQ', cv['pq'], 'tau', cv['best_tau'], 'per fold', cv['per_fold'])"),
    md("## 6. Submission\nSelected components are re-thresholded on the upsampled 2048 map and RLE-encoded; "
       "the validator checks ids, empty/overlapping masks and orientation."),
    code("sub_path = f'runs/{ENS}/test_preds_scorer_v2.csv'\n"
         "if not done(sub_path):\n"
         "    run('scorer.py', '--run', f'runs/{ENS}', '--thr', '0.3', '--v2', '--members', *members, '--predict')\n"
         "run('validate_submission.py', sub_path)\n"
         "pd.read_csv(os.path.join(S2, sub_path)).head()"),
    md("## 7. Qualitative check"),
    code("from IPython.display import Image\n"
         "Image(os.path.join(ROOT, 'report', 'fig_overlay.png'), width=900)"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, os.path.join(ROOT, "pipeline.ipynb"))
print("wrote pipeline.ipynb")
