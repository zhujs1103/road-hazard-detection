from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAME = "litter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RoLID-11K into an Ultralytics YOLO dataset. "
            "Supports existing YOLO labels, VOC XML, and COCO JSON."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--single-class", action="store_true", default=True)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating hard links.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def yolo_label_for_image(image: Path) -> Path | None:
    candidates = [
        image.with_suffix(".txt"),
        image.parent.parent / "labels" / image.with_suffix(".txt").name,
        image.parent.parent / "Labels" / image.with_suffix(".txt").name,
        image.parent.parent / "annotations" / image.with_suffix(".txt").name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def xml_label_for_image(image: Path) -> Path | None:
    candidates = [
        image.with_suffix(".xml"),
        image.parent.parent / "annotations" / image.with_suffix(".xml").name,
        image.parent.parent / "Annotations" / image.with_suffix(".xml").name,
        image.parent.parent / "xmls" / image.with_suffix(".xml").name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_yolo_label(path: Path, single_class: bool) -> list[str]:
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        class_id = 0 if single_class else int(float(parts[0]))
        coords = [float(value) for value in parts[1:5]]
        rows.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in coords))
    return rows


def read_voc_xml(path: Path, image: Path, class_to_id: dict[str, int], single_class: bool) -> list[str]:
    width, height = image_size(image)
    tree = ET.parse(path)
    root = tree.getroot()
    rows: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or DEFAULT_CLASS_NAME
        class_id = 0 if single_class else class_to_id.setdefault(name, len(class_to_id))
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        rows.append(
            f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return rows


def split_from_json_name(path: Path) -> str | None:
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "val" in name or "valid" in name:
        return "val"
    if "test" in name:
        return "test"
    return None


def load_coco_annotations(
    root: Path, single_class: bool
) -> tuple[dict[str, list[str]], dict[str, str], dict[int, str]]:
    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        return {}, {}, {0: DEFAULT_CLASS_NAME}

    label_map: dict[str, list[str]] = {}
    split_map: dict[str, str] = {}
    names: dict[int, str] = {0: DEFAULT_CLASS_NAME}
    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not {"images", "annotations"}.issubset(data):
            continue

        split = split_from_json_name(json_file)
        images = {item["id"]: item for item in data["images"]}
        if split:
            for image_info in data["images"]:
                split_map[Path(image_info["file_name"]).name] = split

        category_to_class: dict[int, int] = {}
        for category in data.get("categories", []):
            if single_class:
                category_to_class[category["id"]] = 0
            else:
                class_id = len(category_to_class)
                category_to_class[category["id"]] = class_id
                names[class_id] = category.get("name", str(class_id))

        for annotation in data["annotations"]:
            image_info = images.get(annotation.get("image_id"))
            if not image_info or "bbox" not in annotation:
                continue
            width = float(image_info["width"])
            height = float(image_info["height"])
            x, y, w, h = [float(value) for value in annotation["bbox"]]
            class_id = category_to_class.get(annotation.get("category_id"), 0)
            row = (
                f"{class_id} {(x + w / 2.0) / width:.8f} {(y + h / 2.0) / height:.8f} "
                f"{w / width:.8f} {h / height:.8f}"
            )
            label_map.setdefault(Path(image_info["file_name"]).name, []).append(row)

    return label_map, split_map, names


def split_name(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "train" in parts or "training" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "test" in parts or "testing" in parts:
        return "test"
    return None


def choose_splits(images: list[Path], seed: int, val_ratio: float, test_ratio: float) -> dict[Path, str]:
    explicit = {image: split_name(image) for image in images}
    if all(value is not None for value in explicit.values()):
        return {image: value or "train" for image, value in explicit.items()}

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    split_map: dict[Path, str] = {}
    for idx, image in enumerate(shuffled):
        if idx < test_count:
            split_map[image] = "test"
        elif idx < test_count + val_count:
            split_map[image] = "val"
        else:
            split_map[image] = "train"
    return split_map


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    raw_root = resolve(args.raw_root)
    out_root = resolve(args.out_root)
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    images = find_images(raw_root)
    if not images:
        raise RuntimeError(f"No images found under {raw_root}")

    coco_labels, coco_split_map, coco_names = load_coco_annotations(raw_root, args.single_class)
    class_to_id: dict[str, int] = {DEFAULT_CLASS_NAME: 0}
    names: dict[int, str] = dict(coco_names)
    if coco_split_map:
        split_map = {
            image: coco_split_map.get(image.name) or split_name(image) or "train"
            for image in images
        }
    else:
        split_map = choose_splits(images, args.seed, args.val_ratio, args.test_ratio)
    counts: Counter[str] = Counter()
    labeled = 0

    for image in images:
        split = split_map[image]
        image_dst = out_root / "images" / split / image.name
        label_dst = out_root / "labels" / split / f"{image.stem}.txt"

        rows = coco_labels.get(image.name)
        if rows is None:
            yolo = yolo_label_for_image(image)
            xml = xml_label_for_image(image)
            if yolo:
                rows = read_yolo_label(yolo, args.single_class)
            elif xml:
                rows = read_voc_xml(xml, image, class_to_id, args.single_class)
            else:
                rows = []

        link_or_copy(image, image_dst, args.copy)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        labeled += bool(rows)
        counts[split] += 1

    if not args.single_class:
        names.update({idx: name for name, idx in class_to_id.items()})

    dataset = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with (out_root / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset, handle, sort_keys=False, allow_unicode=True)

    print(f"Images: {len(images)}")
    print(f"Images with labels: {labeled}")
    print(f"Split counts: {dict(counts)}")
    print(f"Wrote: {out_root / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
