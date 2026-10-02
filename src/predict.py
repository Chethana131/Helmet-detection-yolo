"""Run helmet detection on an image, a folder of images, or a video.

Usage:
    python -m src.predict --source path/to/image.jpg
    python -m src.predict --source path/to/video.mp4 --conf 0.3
    python -m src.predict --source path/to/folder_of_images/
Outputs are written to results/predictions/.
"""
import argparse
from pathlib import Path

import cv2

from src import config
from src.utils import (IMAGE_EXTS, VIDEO_EXTS, detect_image, load_model,
                       process_video, resolve_weights, safety_message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="image, video or folder of images")
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=config.DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=config.DEFAULT_IOU)
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "predictions")
    args = p.parse_args()

    weights, is_custom = resolve_weights(args.weights)
    if not is_custom:
        print("WARNING: models/best.pt not found - using generic COCO weights (no helmet class). Train first!")
    model = load_model(weights)
    args.out.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    if src.is_dir():
        files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    else:
        files = [src]

    for f in files:
        if f.suffix.lower() in VIDEO_EXTS:
            dst = args.out / f"{f.stem}_detected.mp4"
            stats = process_video(model, f, dst, args.conf, args.iou)
            print(f"{f.name}: {stats['frames']} frames, {stats['frames_with_violation']} with a "
                  f"no-helmet detection -> {dst}")
        elif f.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(f))
            annotated, counts, _ = detect_image(model, img, args.conf, args.iou)
            dst = args.out / f"{f.stem}_detected.jpg"
            cv2.imwrite(str(dst), annotated)
            print(f"{f.name}: {dict(counts)} | {safety_message(counts)} -> {dst}")
        else:
            print(f"Skipping unsupported file: {f}")


if __name__ == "__main__":
    main()
