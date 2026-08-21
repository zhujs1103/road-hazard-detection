# Reproduction Report

Date: 2026-04-28

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM
- Driver: 592.01
- Python: 3.10.20 in `bus-detector`
- PyTorch: 2.11.0+cu128
- Torchvision: 0.26.0+cu128
- Ultralytics: 8.4.42
- CUDA available: yes, CUDA runtime 12.8

## Road Damage Detection: ORDDC

Source: `third_party/orddc2024-main`

Changes needed for the local RTX 50-series environment:

- Kept PyTorch 2.11.0+cu128 instead of the repository-pinned PyTorch 2.2.2.
- Patched YOLOv5 checkpoint loading in
  `third_party/orddc2024-main/orddc2024/predictors/yolov5/models/experimental.py`
  to pass `weights_only=False`, because PyTorch 2.6+ changed the default
  `torch.load` behavior.

Smoke command:

```powershell
python scripts/run_orddc.py --phase 1 --images third_party\orddc2024-main\train_scripts\data\sample\test --output outputs\orddc_phase1_sample.csv
python scripts/visualize_orddc_csv.py --images third_party\orddc2024-main\train_scripts\data\sample\test --csv outputs\orddc_phase1_sample.csv --output-dir outputs\orddc_phase1_sample_vis_clean --nms-iou 0.3 --max-boxes 20
```

Outputs:

- CSV: `outputs/orddc_phase1_sample.csv`
- Visualized images: `outputs/orddc_phase1_sample_vis_clean/`
- Official model bundle: `third_party/orddc2024-main/models_ph1/`

Result: sample inference completed successfully on GPU.

## Roadside Litter Detection: RoLID-11K

Source: `third_party/RoLID-11K-main`

Official data downloaded from the Google Drive folder referenced in the
repository README. The downloaded files are stored under:

```text
data/raw/rolid-11k/
```

The dataset uses COCO JSON annotations:

- `training.json`: 7990 images
- `validation.json`: 1201 images
- `testing.json`: 2373 images
- Effective detection class: `litter`

Conversion command:

```powershell
python scripts/prepare_rolid_yolo.py --raw-root data\raw\rolid-11k --out-root data\processed\rolid_yolo
```

Converted YOLO dataset:

```text
data/processed/rolid_yolo/dataset.yaml
```

Smoke train command:

```powershell
python scripts/train_litter_yolo.py --data data\processed\rolid_yolo\dataset.yaml --model yolo11n.pt --imgsz 640 --epochs 1 --batch 4 --device 0 --workers 0 --name smoke
```

Smoke result after 1 epoch:

| Metric | Value |
| --- | ---: |
| Precision | 0.51581 |
| Recall | 0.37513 |
| mAP50 | 0.34863 |
| mAP50-95 | 0.09786 |

Outputs:

- Weights: `outputs/rolid_yolo/smoke/weights/best.pt`
- Training curves: `outputs/rolid_yolo/smoke/results.png`
- Validation visualizations: `outputs/rolid_yolo/smoke/val_batch*_pred.jpg`

Recommended full baseline:

```powershell
python scripts/train_litter_yolo.py --data data\processed\rolid_yolo\dataset.yaml --model yolo11n.pt --imgsz 960 --epochs 80 --batch 8 --device 0 --workers 4 --name yolo11n_960_e80
```

If VRAM is tight, reduce `--batch` to 4.

## RoLID-11K Full YOLO11n Baseline

Training command completed:

```powershell
python scripts\train_litter_yolo.py --data data\processed\rolid_yolo\dataset.yaml --model yolo11n.pt --imgsz 960 --epochs 80 --batch 8 --device 0 --workers 4 --name yolo11n_960_e80
```

Validation result reported by Ultralytics after loading `best.pt`:

| Split | Images | Instances | Precision | Recall | mAP50 | mAP50-95 | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 1201 | 2093 | 0.734 | 0.697 | 0.747 | 0.302 | 1.7 ms/image |

Official test split was evaluated by generating a temporary validation YAML
that points `val` to `images/test`, because Ultralytics 8.4.42 did not honor
`split=test` directly in this local run.

Test command:

```powershell
python scripts/train_litter_yolo.py --mode val --weights outputs\rolid_yolo\yolo11n_960_e80\weights\best.pt --data data\processed\rolid_yolo\dataset.yaml --imgsz 960 --batch 8 --device 0 --workers 0 --split test --name yolo11n_960_e80_test
```

Test result:

| Split | Images | Instances | Precision | Recall | mAP50 | mAP50-95 | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 2373 | 4188 | 0.757 | 0.668 | 0.732 | 0.289 | 2.9 ms/image |

Outputs:

- Full baseline weights: `outputs/rolid_yolo/yolo11n_960_e80/weights/best.pt`
- Test curves and batch predictions: `outputs/rolid_yolo/yolo11n_960_e80_test/`

## Real-World ORDDC Video Check

Run folder:

```text
outputs/orddc_phase1_full_2s_20260525/
```

Input video: `0.mp4`, sampled every 2 seconds.

Outputs:

- Sampled frames: `frames_2s/`, 196 images
- Visualized NMS results: `visualized_nms/`, 196 images
- Raw ensemble CSV: `orddc_phase1_ensemble_full_2s.csv`, 196 rows
- Manual false-positive notes: `manual_review_false_positives.csv`, 15 reviewed examples
- Local run report: `README_report.md`

Summary:

- 88 of 196 sampled frames contain at least one road-damage candidate.
- Raw candidate boxes before visualization filtering: 6902.
- Raw candidate class counts: D00 5555, D10 1147, D20 70, D40 130.
- Manual review found repeated false positives from shadows, road markings,
  zebra-crossing paint gaps, wet stains, black liquid or dirt traces, curb edges,
  and video compression or low-pixel road texture.

Conclusion: these outputs should be treated as road-damage candidates, not as
ground-truth damage counts. Before engineering use for statistics or alerts, this
video needs scene-specific human labels, temporal merging, spatial deduplication,
false-positive filtering, and targeted fine-tuning with hard negatives from the
observed failure modes.
