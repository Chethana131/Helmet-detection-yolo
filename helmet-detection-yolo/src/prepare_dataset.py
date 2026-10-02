"""Convert the Kaggle Hard Hat dataset (Pascal VOC XML) into YOLO format.

VOC box  : (xmin, ymin, xmax, ymax) in pixels, one XML file per image.
YOLO box : (class_id, x_center, y_center, width, height), all normalised to
           0..1 by the image size, one .txt file per image.

The script also creates a reproducible train / val / test split.

Usage:
    python -m src.prepare_dataset
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from src import config

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_voc(xml_path: Path):
    """Return (img_w, img_h, [(class_name, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.find("width").text)), int(float(size.find("height").text))
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return w, h, boxes


def voc_to_yolo_lines(w, h, boxes):
    """Convert VOC boxes to YOLO text lines, dropping unknown/degenerate boxes."""
    lines = []
    for name, xmin, ymin, xmax, ymax in boxes:
        if name not in config.CLASS_NAMES:
            continue
        # Clip to the image so that normalised values stay inside [0, 1].
        xmin, xmax = max(0.0, xmin), min(float(w), xmax)
        ymin, ymax = max(0.0, ymin), min(float(h), ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        xc, yc = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        cid = config.CLASS_NAMES.index(name)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def write_data_yaml(out_dir: Path):
    """Write the data.yaml file Ultralytics needs (plain text, no PyYAML needed)."""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(config.CLASS_NAMES))
    text = (
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n"
    )
    (out_dir / "data.yaml").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=config.RAW_DIR, help="folder with the unzipped Kaggle data")
    parser.add_argument("--out", type=Path, default=config.YOLO_DIR, help="output folder (YOLO format)")
    args = parser.parse_args()

    # 1) Find every XML annotation and every image, matched by file name stem.
    xmls = sorted(args.raw.rglob("*.xml"))
    images = {p.stem: p for p in args.raw.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    pairs = [(x, images[x.stem]) for x in xmls if x.stem in images]
    if not pairs:
        raise SystemExit(
            f"No (image, xml) pairs found under {args.raw}.\n"
            "Download the dataset first - see the README, section 'Dataset'."
        )
    print(f"Found {len(pairs)} annotated images.")

    # 2) Shuffle with a fixed seed -> the split is identical on every run.
    random.Random(config.SEED).shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(n * config.TRAIN_FRAC), int(n * config.VAL_FRAC)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    # 3) Rebuild the output folder from scratch (it is generated data).
    if args.out.exists():
        shutil.rmtree(args.out)
    class_counts = {c: 0 for c in config.CLASS_NAMES}
    for split, items in splits.items():
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        for xml_path, img_path in items:
            w, h, boxes = parse_voc(xml_path)
            lines = voc_to_yolo_lines(w, h, boxes)
            for ln in lines:
                class_counts[config.CLASS_NAMES[int(ln.split()[0])]] += 1
            shutil.copy2(img_path, args.out / "images" / split / img_path.name)
            (args.out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines))
        print(f"  {split:5s}: {len(items)} images")

    write_data_yaml(args.out)
    print("Box counts per class:", class_counts)
    print(f"Done. Dataset config written to {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
