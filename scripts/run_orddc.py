from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDDC_ROOT = ROOT / "third_party" / "orddc2024-main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official ORDDC inference.")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not ORDDC_ROOT.exists():
        raise FileNotFoundError(f"Missing ORDDC repo: {ORDDC_ROOT}")

    images = args.images
    if not images.is_absolute():
        images = ROOT / images
    if not images.exists():
        raise FileNotFoundError(f"Missing image directory: {images}")

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    script = ORDDC_ROOT / f"inference_script_v2_Phase{args.phase}.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing inference script: {script}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ORDDC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, str(script), str(images), str(output)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ORDDC_ROOT), env=env, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
