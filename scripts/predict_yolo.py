from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction and save visualized outputs.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    weights = resolve(args.weights)
    source = resolve(args.source)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
