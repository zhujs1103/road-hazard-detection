from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or validate a YOLO baseline on RoLID-11K.")
    parser.add_argument("--mode", choices=["train", "val"], default="train")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt", help="Model for training.")
    parser.add_argument("--weights", type=Path, help="Weights for validation.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split for validation.")
    parser.add_argument("--project", type=Path, default=ROOT / "outputs" / "rolid_yolo")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def validation_data_file(data: Path, split: str, project: Path) -> Path:
    if split == "val":
        return data

    dataset = yaml.safe_load(data.read_text(encoding="utf-8"))
    test_path = dataset.get("test")
    if not test_path:
        raise ValueError(f"{data} does not define a test split")

    dataset["val"] = test_path
    out_dir = project / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{data.stem}_{split}_as_val.yaml"
    out_file.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return out_file


def main() -> int:
    args = parse_args()
    data = resolve(args.data)
    project = resolve(args.project)

    if args.mode == "train":
        model = YOLO(args.model)
        name = args.name or "train"
        model.train(
            data=str(data),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=name,
            exist_ok=True,
            pretrained=True,
            close_mosaic=10,
        )
    else:
        if not args.weights:
            raise ValueError("--weights is required for --mode val")
        model = YOLO(str(resolve(args.weights)))
        name = args.name or "val"
        val_data = validation_data_file(data, args.split, project)
        model.val(
            data=str(val_data),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=name,
            exist_ok=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
