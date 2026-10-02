"""Central configuration for the helmet detection project.

Every path and default hyper-parameter lives here so that the other scripts
stay short and there is exactly one place to change things.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded Kaggle files (Pascal VOC format)
YOLO_DIR = DATA_DIR / "yolo"        # converted dataset (YOLO format)
DATA_YAML = YOLO_DIR / "data.yaml"  # dataset description read by Ultralytics
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"            # Ultralytics training output
BEST_WEIGHTS = MODELS_DIR / "best.pt"

# --------------------------------------------------------------- dataset ---
# Kaggle "Hard Hat Detection" (andrewmvd/hard-hat-detection): 5000 images,
# Pascal VOC XML annotations with three classes.
KAGGLE_DATASET = "andrewmvd/hard-hat-detection"
# The ORDER matters: the position in this list is the class id used by YOLO.
CLASS_NAMES = ["helmet", "head", "person"]
# "head" = a bare head, i.e. a worker WITHOUT a helmet -> a safety violation.
VIOLATION_CLASS = "head"

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.20   # remaining 10 % becomes the test set

# -------------------------------------------------------------- defaults ---
DEFAULT_BASE_MODEL = "yolov8n.pt"   # smallest/fastest YOLOv8; auto-downloaded
DEFAULT_EPOCHS = 30
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_CONF = 0.25                 # confidence threshold for inference
DEFAULT_IOU = 0.45                  # IoU threshold for NMS at inference
