from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    0: "D00 longitudinal crack",
    1: "D10 transverse crack",
    2: "D20 alligator crack",
    3: "D40 pothole",
}
COLORS = {
    0: (255, 80, 80),
    1: (80, 180, 255),
    2: (80, 220, 120),
    3: (240, 120, 255),
}


def normalize_class_id(class_id: int) -> int:
    # Official ORDDC submission CSV uses 1-based class ids, while local YOLO
    # label files are 0-based.
    return class_id - 1 if 1 <= class_id <= 4 else class_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw ORDDC submission CSV boxes.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.5, help="Visualization-only duplicate box removal.")
    parser.add_argument("--max-boxes", type=int, default=80, help="Maximum boxes drawn per image after NMS.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(image, (x, y0), (x + tw + 6, y0 + th + baseline + 4), color, -1)
    cv2.putText(image, text, (x + 3, y0 + th + 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def dedupe_boxes(
    boxes: list[tuple[int, int, int, int, int]], nms_iou: float, max_boxes: int
) -> list[tuple[int, int, int, int, int]]:
    kept: list[tuple[int, int, int, int, int]] = []
    # Submission CSV has no confidence score. Prefer larger boxes first for display.
    sorted_boxes = sorted(boxes, key=lambda item: (item[3] - item[1]) * (item[4] - item[2]), reverse=True)
    for item in sorted_boxes:
        class_id, x1, y1, x2, y2 = item
        if all(class_id != old[0] or iou((x1, y1, x2, y2), old[1:]) < nms_iou for old in kept):
            kept.append(item)
        if len(kept) >= max_boxes:
            break
    return kept


def main() -> int:
    args = parse_args()
    image_dir = resolve(args.images)
    csv_path = resolve(args.csv)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    count = 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            image_name = row[0].lstrip("\ufeff")
            predictions = row[1] if len(row) > 1 else ""
            image_path = image_dir / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARN] missing image: {image_path}")
                continue

            tokens = predictions.split()
            boxes: list[tuple[int, int, int, int, int]] = []
            for idx in range(0, len(tokens), 5):
                try:
                    class_id, x1, y1, x2, y2 = map(int, tokens[idx : idx + 5])
                except ValueError:
                    continue
                class_id = normalize_class_id(class_id)
                boxes.append((class_id, x1, y1, x2, y2))

            for class_id, x1, y1, x2, y2 in dedupe_boxes(boxes, args.nms_iou, args.max_boxes):
                color = COLORS.get(class_id, (255, 255, 255))
                label = CLASS_NAMES.get(class_id, str(class_id))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                draw_label(image, label, x1, y1, color)

            cv2.imwrite(str(output_dir / image_name), image)
            count += 1

    print(f"Wrote {count} visualized images to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
