"""Evaluate trained weights on the held-out TEST split.

Calculates precision, recall, mAP@0.5 and mAP@0.5:0.95 (overall and per class),
and saves them to results/metrics.json and results/metrics.md.
Ultralytics also saves the PR curve and confusion matrix (results/test_eval/).

Usage:
    python -m src.evaluate
    python -m src.evaluate --weights runs/helmet_yolov8/weights/best.pt
"""
import argparse
import json

from ultralytics import YOLO

from src import config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=str(config.BEST_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model = YOLO(args.weights)
    val_kwargs = dict(
        data=str(config.DATA_YAML),
        split=args.split,
        imgsz=args.imgsz,
        project=str(config.RESULTS_DIR),
        name=f"{args.split}_eval",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)   # conf=0.001 by default: standard for mAP
    box = metrics.box

    overall = {
        "precision": float(box.mp),      # mean precision over classes
        "recall": float(box.mr),         # mean recall over classes
        "mAP@0.5": float(box.map50),
        "mAP@0.5:0.95": float(box.map),
    }
    per_class = {}
    for i, class_id in enumerate(box.ap_class_index):
        pr, rc = float(box.p[i]), float(box.r[i])
        per_class[metrics.names[int(class_id)]] = {
            "precision": pr,
            "recall": rc,
            "f1": 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0,
            "mAP@0.5": float(box.ap50[i]),
            "mAP@0.5:0.95": float(box.ap[i]),
        }

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = {"split": args.split, "weights": str(args.weights), "overall": overall, "per_class": per_class}
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))

    # A README-ready markdown table.
    lines = [f"### Results on the `{args.split}` split", "",
             "| Class | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|---|"]
    for name, m in per_class.items():
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                     f"{m['mAP@0.5']:.3f} | {m['mAP@0.5:0.95']:.3f} |")
    lines.append(f"| **all** | **{overall['precision']:.3f}** | **{overall['recall']:.3f}** | - | "
                 f"**{overall['mAP@0.5']:.3f}** | **{overall['mAP@0.5:0.95']:.3f}** |")
    (config.RESULTS_DIR / "metrics.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved: {config.RESULTS_DIR / 'metrics.json'}, {config.RESULTS_DIR / 'metrics.md'}")
    print(f"Plots (PR curve, confusion matrix): {config.RESULTS_DIR / (args.split + '_eval')}")


if __name__ == "__main__":
    main()
