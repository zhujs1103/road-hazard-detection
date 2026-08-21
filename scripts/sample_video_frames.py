from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample frames from a video at a fixed time interval.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--prefix", default="frame")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    video = resolve(args.video)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    step = max(1, int(round(args.interval_sec * fps)))

    start_frame = max(0, int(round(args.start_sec * fps)))
    end_frame = frame_count if args.end_sec is None else min(frame_count, int(round(args.end_sec * fps)))

    saved = 0
    frame_idx = start_frame
    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps
        out_name = f"{args.prefix}_{saved:04d}_{timestamp:07.2f}s.jpg"
        cv2.imwrite(str(output_dir / out_name), frame)
        saved += 1
        frame_idx += step

    cap.release()
    print(f"video={video}")
    print(f"fps={fps:.3f}")
    print(f"frames={frame_count}")
    print(f"duration_sec={duration:.2f}")
    print(f"sample_interval_sec={args.interval_sec}")
    print(f"start_sec={args.start_sec}")
    print(f"end_sec={args.end_sec if args.end_sec is not None else duration:.2f}")
    print(f"saved={saved}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
