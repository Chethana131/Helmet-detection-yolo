"""Fine-tune a pretrained YOLOv8 model on the helmet dataset.

Usage:
    python -m src.train                          # sensible defaults
    python -m src.train --epochs 50 --model yolov8s.pt
    python -m src.train --epochs 1 --fraction 0.05   # quick smoke test
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from src import config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.DEFAULT_BASE_MODEL, help="pretrained checkpoint to start from")
    p.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=config.DEFAULT_BATCH)
    p.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu; default = auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of the training set to use")
    p.add_argument("--name", default="helmet_yolov8")
    args = p.parse_args()

    if not config.DATA_YAML.exists():
        raise SystemExit("data.yaml not found. Run:  python -m src.prepare_dataset")

    # Transfer learning: start from COCO-pretrained weights, then fine-tune.
    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(config.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        seed=config.SEED,
        project=str(config.RUNS_DIR),
        name=args.name,
        exist_ok=True,
        plots=True,          # saves curves + confusion matrix to the run folder
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    # Copy the best checkpoint (highest validation fitness) to models/best.pt.
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, config.BEST_WEIGHTS)
    print(f"\nBest weights copied to {config.BEST_WEIGHTS}")
    print("Next step:  python -m src.evaluate")


if __name__ == "__main__":
    main()
