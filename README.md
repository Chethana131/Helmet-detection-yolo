# ⛑️ Helmet Detection with YOLOv8

Real-time detection of **helmets**, **bare heads (no helmet)** and **persons** in images and video, built by fine-tuning Ultralytics YOLOv8 on the Kaggle *Hard Hat Detection* dataset. Includes training, evaluation (precision, recall, mAP), image/video inference and a Streamlit demo that flags safety violations.

> **Results** (fill in after running `python -m src.evaluate`)
>
> | Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
> |---|---|---|---|---|
> | all | _from results/metrics.md_ | | | |

![demo](docs/demo.gif) <!-- add your own GIF, see "Screenshots to capture" -->

## Features
- Dataset converter: Pascal VOC XML → YOLO format with a reproducible 70/20/10 split
- Transfer learning from COCO-pretrained YOLOv8
- Evaluation on a held-out test set: precision, recall, mAP@0.5, mAP@0.5:0.95, per class
- Image, folder and video inference from the command line
- Streamlit app with confidence / IoU sliders and a "worker without helmet" warning

## Project structure
```
helmet-detection-yolo/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py            # paths, class names, defaults
│   ├── prepare_dataset.py   # VOC XML -> YOLO format + train/val/test split
│   ├── train.py             # fine-tune YOLOv8
│   ├── evaluate.py          # precision / recall / mAP on the test split
│   ├── predict.py           # CLI inference (image / folder / video)
│   └── utils.py             # shared detection helpers
├── app/streamlit_app.py     # web demo (image, camera, video, metrics dashboard)
├── assets/background.svg    # app background image (replace with your own background.jpg if you like)
├── .streamlit/config.toml   # dark theme used by the app
├── docs/STUDY_GUIDE.md      # concepts + interview Q&A
├── data/                    # (git-ignored) raw + converted dataset
├── models/                  # (git-ignored) best.pt
└── results/                 # metrics + plots
```

## Dataset
**Hard Hat Detection** by Larxel on Kaggle: 5,000 images, Pascal VOC annotations, 3 classes:
`helmet`, `head` (bare head = no helmet), `person`.
<https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection>

**Option A: Kaggle CLI**
1. Kaggle → Settings → *Create New Token*; save `kaggle.json` to `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`). Never commit it.
2. ```bash
   kaggle datasets download -d andrewmvd/hard-hat-detection -p data/raw --unzip
   ```

**Option B: manual**: click *Download* on the Kaggle page and unzip so that `data/raw/` contains the `images/` and `annotations/` folders.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.prepare_dataset          # 1. convert + split
python -m src.train                    # 2. train (30 epochs, yolov8n)
python -m src.evaluate                 # 3. metrics on the test split
python -m src.predict --source some_image.jpg
python -m src.predict --source some_video.mp4
streamlit run app/streamlit_app.py     # 4. web demo
```
Quick smoke test (a few minutes, checks that everything is wired up):
`python -m src.train --epochs 1 --fraction 0.05 --batch 8`

**No GPU?** Training on CPU is slow. Use a free Google Colab GPU: *Runtime → Change runtime type → T4 GPU*, clone your repo, `pip install -r requirements.txt`, run the same commands (use `--device 0`), then download `models/best.pt` and `results/`.

Useful options: `--model yolov8s.pt` (bigger, more accurate), `--epochs 50`, `--imgsz 640`, `--batch 32`, `--device 0`.

## Streamlit app
Four tabs: **Image detection** (upload or camera, violation banner, downloadable annotated image), **Video detection** (per-frame violation stats), **Model performance** (your real `results/metrics.json` plus the evaluation plots) and **How it works**.
Custom background: put a `background.jpg` (or `.png` / `.webp`, ideally under 1.5 MB) in `assets/`; it is used automatically instead of the bundled `background.svg`. Run the app from the repository root so the dark theme in `.streamlit/config.toml` is picked up.

## Architecture / workflow
```
Kaggle images + VOC XML
        │  prepare_dataset.py   (parse XML, normalise boxes, split 70/20/10)
        ▼
data/yolo/{images,labels}/{train,val,test} + data.yaml
        │  train.py             (YOLOv8 pretrained on COCO → fine-tune)
        ▼
runs/helmet_yolov8/ (curves, val metrics)  →  models/best.pt
        │  evaluate.py          (test split → P, R, mAP, PR curve, confusion matrix)
        ▼
results/metrics.json, metrics.md, test_eval/
        │  predict.py / streamlit_app.py
        ▼
annotated image / video + "X workers without helmet" warning
```
Why three splits? **train** updates the weights, **val** picks the best epoch (`best.pt`) and drives early stopping, **test** is touched once at the end for an honest estimate.

## Concepts explained
**Object detection**: unlike classification (one label per image), detection finds *every* object, tells you *what* it is and *where* it is.

**Bounding box**: a rectangle around an object. YOLO stores it as `(x_center, y_center, width, height)` normalised to 0–1 by the image size, so labels do not depend on image resolution.

**YOLO ("You Only Look Once")**: a single neural network looks at the whole image once and directly predicts boxes + classes. That is much faster than older two-stage detectors (e.g. Faster R-CNN) that first propose regions and then classify them, which is why YOLO suits real-time video. YOLOv8 is *anchor-free* (it predicts box centres directly instead of adjusting preset anchor shapes) and uses a decoupled head (separate branches for box and class).

**Transfer learning**: we start from weights trained on COCO (80 everyday classes). Early layers already detect edges, textures and shapes, so we only need to adapt to helmets, which needs far less data and time than training from scratch.

**IoU (Intersection over Union)**: `overlap area / union area` of a predicted and a true box (0 = no overlap, 1 = identical). A prediction counts as correct (a *true positive*) only if IoU is above a threshold (0.5 for mAP@0.5).

**Confidence score**: how sure the model is that a box contains a certain class (0–1). At inference, boxes below `--conf` are discarded: higher threshold → fewer false alarms but more missed objects.

**NMS (non-maximum suppression)**: the network proposes many overlapping boxes for one object; NMS keeps the highest-confidence one and removes others that overlap it by more than the NMS IoU threshold.

**Precision** = TP / (TP + FP): of the boxes the model drew, how many were right. **Recall** = TP / (TP + FN): of all real objects, how many it found. In safety, missing a bare head (low recall) is the dangerous error.

**Average Precision (AP) and mAP**: as the confidence threshold sweeps from 1 to 0 you get a precision-recall curve; AP is the area under it for one class. **mAP** is the mean of AP over classes. **mAP@0.5** uses IoU ≥ 0.5; **mAP@0.5:0.95** averages over IoU thresholds 0.5, 0.55 … 0.95 (the COCO standard) and rewards tightly fitting boxes.

**Data augmentation**: Ultralytics automatically applies mosaic, flips, HSV colour jitter and scaling during training to reduce overfitting.

## Results
Paste the table from `results/metrics.md` here, plus `results/test_eval/PR_curve.png` and `confusion_matrix.png`. Do not quote numbers you did not generate.

## Screenshots / GIFs to capture
1. Streamlit app on an image with detections and the red "WITHOUT a helmet" banner
2. A 5–10 s GIF of the video tab (screen-record, convert to GIF)
3. `results/test_eval/PR_curve.png`
4. `results/test_eval/confusion_matrix_normalized.png`
5. `runs/helmet_yolov8/results.png` (loss and mAP curves per epoch)
6. A grid of `val_batch0_pred.png` from the run folder
7. Terminal screenshot of the evaluate output table
Save to `docs/` and embed them in this README.

## Limitations & future work
- Trained on one dataset (construction scenes, mostly daylight); may drop on night, fog or unusual helmet colours
- `head` could include caps/hoods; the label means "no helmet", not "no head cover"
- Next: try `yolov8s/m`, export to ONNX/TensorRT, add object tracking so each worker is counted once across frames

## Resume bullet
> Fine-tuned YOLOv8 on 5,000 construction-site images to detect helmets / bare heads in images and video; achieved **[X] mAP@0.5** and **[Y] recall** on a held-out test set; built a Streamlit app that flags safety violations in real time. *(Tech: Python, Ultralytics YOLO, OpenCV, Streamlit)*

## License / credits
Dataset © its Kaggle author; check the dataset page for its license. Model: Ultralytics YOLOv8 (AGPL-3.0). Keep this in mind if you reuse the code commercially.
