# Bus Detector Reproduction

This workspace contains reproducible baselines for the first two project tasks:

1. Road damage detection: ORDDC/RDD style crack and pothole detection.
2. Roadside litter detection: RoLID-11K small-object litter detection.

The public repository provides thin wrappers under `scripts/`. It expects the
upstream projects and datasets under `third_party/` and `data/`, but does not
redistribute them.

## Upstream assets

- Clone [USC-InfoLab/orddc2024](https://github.com/USC-InfoLab/orddc2024) to
  `third_party/orddc2024-main/` (or adjust the wrapper path).
- Download RoLID-11K from the official location linked by its authors and place
  it under `data/raw/rolid-11k/`.
- Review the upstream code and dataset licenses before use or redistribution.

## Environment

Use the `bus-detector` conda environment with Python 3.10 and CUDA 12.8 PyTorch
for RTX 50-series GPUs.

```powershell
conda activate bus-detector
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu-cu128.txt
python scripts/check_env.py
```

If Anaconda is not on `PATH`, activate the environment from Anaconda Prompt or
replace `python` with the full path to your environment interpreter.

Do not install `third_party/orddc2024-main/requirements.txt` blindly on an RTX
50-series laptop. It pins an older PyTorch build.

## Road Damage: ORDDC

Run the official ORDDC Phase 1 ensemble on the included sample images:

```powershell
python scripts/run_orddc.py --phase 1 --images third_party/orddc2024-main/train_scripts/data/sample/test --output outputs/orddc_phase1_sample.csv
python scripts/visualize_orddc_csv.py --images third_party/orddc2024-main/train_scripts/data/sample/test --csv outputs/orddc_phase1_sample.csv --output-dir outputs/orddc_phase1_sample_vis_clean --nms-iou 0.3 --max-boxes 20
```

The first command downloads the official model bundle through `gdown` if the
third-party script cannot find it. The second command renders detection boxes so
the result can be inspected visually.

For Phase 2:

```powershell
python scripts/run_orddc.py --phase 2 --images third_party/orddc2024-main/train_scripts/data/sample/test --output outputs/orddc_phase2_sample.csv
```

## Roadside Litter: RoLID-11K

Download RoLID-11K from the official Google Drive or Kaggle mirror into:

```text
data/raw/rolid-11k/
```

The current workspace already has the official Google Drive data downloaded and
extracted there.

Then convert it to a standard Ultralytics YOLO layout:

```powershell
python scripts/prepare_rolid_yolo.py --raw-root data/raw/rolid-11k --out-root data/processed/rolid_yolo
```

Train a YOLO baseline:

```powershell
python scripts/train_litter_yolo.py --data data/processed/rolid_yolo/dataset.yaml --model yolo11n.pt --imgsz 960 --epochs 80 --batch 8 --device 0
```

For a quick pipeline smoke test:

```powershell
python scripts/train_litter_yolo.py --data data/processed/rolid_yolo/dataset.yaml --model yolo11n.pt --imgsz 640 --epochs 1 --batch 4 --device 0 --workers 0 --name smoke
```

Evaluate and predict:

```powershell
python scripts/train_litter_yolo.py --mode val --weights outputs/rolid_yolo/train/weights/best.pt --data data/processed/rolid_yolo/dataset.yaml --imgsz 960 --device 0
python scripts/train_litter_yolo.py --mode val --weights outputs/rolid_yolo/yolo11n_960_e80/weights/best.pt --data data/processed/rolid_yolo/dataset.yaml --imgsz 960 --batch 8 --device 0 --workers 0 --split test --name yolo11n_960_e80_test
python scripts/predict_yolo.py --weights outputs/rolid_yolo/train/weights/best.pt --source data/processed/rolid_yolo/images/val --output outputs/rolid_predictions --imgsz 960 --device 0
```

## Expected Starting Points

- ORDDC: official ensemble and sample inference are the closest available
  reproduction of the researched road damage solution.
- RoLID-11K: the official repository currently publishes data, not training
  code. The provided YOLO baseline reproduces the practical real-time detector
  line from the paper and creates a measurable baseline for project use.

## License

The original code in this curated repository is released under the MIT License.
Upstream projects, model weights and datasets remain subject to their own
licenses and are not redistributed here.
