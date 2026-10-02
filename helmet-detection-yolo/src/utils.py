"""Shared helpers used by predict.py and the Streamlit app."""
import subprocess
from collections import Counter
from pathlib import Path

import cv2

from src import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def resolve_weights(path=None):
    """Return (weights_path, is_custom). Falls back to generic COCO weights."""
    if path and Path(path).exists():
        return str(path), True
    if config.BEST_WEIGHTS.exists():
        return str(config.BEST_WEIGHTS), True
    return config.DEFAULT_BASE_MODEL, False  # generic model: will NOT know helmets


def load_model(weights):
    from ultralytics import YOLO  # imported lazily so the module loads quickly
    return YOLO(weights)


def detect_image(model, img_bgr, conf=config.DEFAULT_CONF, iou=config.DEFAULT_IOU):
    """Run detection on one BGR image.

    Returns (annotated_bgr_image, Counter of class names, ultralytics Result).
    """
    result = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)[0]
    counts = Counter(model.names[int(c)] for c in result.boxes.cls)
    return result.plot(), counts, result


def safety_message(counts):
    """Turn class counts into a human readable safety summary."""
    ok, bad = counts.get("helmet", 0), counts.get(config.VIOLATION_CLASS, 0)
    if bad:
        return f"WARNING: {bad} worker(s) detected WITHOUT a helmet ({ok} with helmet)."
    return f"All clear: {ok} helmet(s) detected, no bare heads."


def process_video(model, src, dst, conf=config.DEFAULT_CONF, iou=config.DEFAULT_IOU, progress_cb=None):
    """Annotate every frame of a video and write it to `dst` (mp4).

    Returns a dict of simple statistics.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames = violation_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, counts, _ = detect_image(model, frame, conf, iou)
        writer.write(annotated)
        frames += 1
        violation_frames += int(counts.get(config.VIOLATION_CLASS, 0) > 0)
        if progress_cb:
            progress_cb(min(frames / total, 1.0))
    cap.release()
    writer.release()
    return {"frames": frames, "frames_with_violation": violation_frames, "fps": fps}


def to_browser_mp4(src, dst):
    """Re-encode to H.264 so browsers can play it. Returns True on success."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([exe, "-y", "-i", str(src), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(dst)],
                       check=True, capture_output=True)
        return True
    except Exception:
        return False
