# Study guide: Helmet Detection (beginner → interview)

## Level 1: Beginner (understand the idea)
1. Classification answers "what is in this image?"; detection answers "what, and where?".
2. A detector outputs a list of `(box, class, confidence)`.
3. Our three classes: `helmet`, `head` (no helmet = violation), `person`.
4. Pipeline: convert data → train → evaluate → predict → demo. Be able to walk through each script in `src/` in one sentence.

## Level 2: Intermediate (understand the code)
Read in this order and be able to explain every function:
- `prepare_dataset.py`: `parse_voc` (read XML), `voc_to_yolo_lines` (pixel corners → normalised centre/width/height, with clipping), the seeded shuffle that makes the split reproducible.
- `train.py`: `YOLO('yolov8n.pt')` loads pretrained weights, `.train()` fine-tunes. `patience` = early stopping, `seed` = reproducibility, `best.pt` = checkpoint with best validation fitness (a weighted mix of mAP@0.5 and mAP@0.5:0.95).
- `evaluate.py`: `model.val(split='test')` returns `metrics.box` with `mp` (mean precision), `mr`, `map50`, `map`, and per-class arrays `p, r, ap50, ap`. Validation uses `conf=0.001` on purpose: to draw the full PR curve you need low-confidence boxes too.
- `utils.py`: `detect_image` calls `model.predict(...)`; `result.boxes.cls/conf/xyxy` hold the detections; `process_video` loops over frames with OpenCV; `to_browser_mp4` re-encodes to H.264 because browsers cannot play OpenCV's `mp4v`.

Exercises: change `--conf` from 0.25 to 0.6 and watch what disappears; print `result.boxes.xyxy` for an image; hand-compute IoU of two boxes.

## Level 3: Interview level (be able to defend decisions)
- Why YOLOv8n? Smallest/fastest; good accuracy/speed trade-off for a portfolio and real-time use. Mention you would try `s`/`m` if accuracy matters more than speed.
- Why a fixed seed and a separate test split? Reproducibility and an unbiased final estimate.
- What would you do for night or rainy scenes? Collect/label that data, stronger augmentation (brightness, blur), re-evaluate per condition.
- How would you deploy? Export to ONNX/TensorRT, run on an edge device or behind a FastAPI endpoint, add tracking, log alerts.

## 12 interview questions and answers
**1. What is YOLO and why is it fast?**
It is a single-stage detector: one forward pass predicts all boxes and classes directly, instead of first proposing regions and then classifying each (two-stage detectors like Faster R-CNN). Fewer stages means lower latency, so it works on video.

**2. Explain IoU and where it is used in your project.**
IoU = intersection area / union area of two boxes. It decides whether a prediction is a true positive during evaluation (≥0.5 for mAP@0.5) and it is the overlap threshold in NMS.

**3. What are precision and recall, and which matters more here?**
Precision = TP/(TP+FP), recall = TP/(TP+FN). For safety monitoring recall on the `head` class matters most: a missed bare head is a missed violation. Too many false alarms (low precision) causes alert fatigue, so I tune the confidence threshold for the trade-off.

**4. What is mAP and the difference between mAP@0.5 and mAP@0.5:0.95?**
AP is the area under the precision-recall curve of one class; mAP averages over classes. mAP@0.5 counts a detection as correct at IoU ≥ 0.5; mAP@0.5:0.95 averages over IoU thresholds from 0.5 to 0.95, so it also measures how precisely the boxes fit and is always lower.

**4b. Why can mAP be high while the demo still looks wrong?**
mAP averages over all confidence thresholds, but the demo uses one threshold (0.25). Also the test set may differ from your own photos (domain shift).

**5. What is non-maximum suppression?**
The model predicts many overlapping boxes for one object. NMS keeps the highest-confidence box and removes others whose IoU with it exceeds a threshold, so each object is reported once. Too low an IoU threshold can merge two nearby workers into one.

**6. Why transfer learning?**
Pretrained COCO weights already encode edges, shapes and textures. Fine-tuning needs less data, converges faster and usually generalises better than random initialisation on a 5k-image dataset.

**7. How did you split data and why?**
70/20/10 train/val/test with a fixed seed. Train updates weights, val selects the best epoch and early-stops, test is evaluated once for an unbiased number. Caveat: the images are random frames from the same source, so near-duplicate scenes could leak between splits; a stricter split would group by scene.

**8. Explain the label format conversion.**
VOC stores pixel corners `(xmin, ymin, xmax, ymax)` per image in XML. YOLO wants `class x_center y_center w h` normalised by image width/height. I compute centre = average of corners / size, w/h = difference / size, and clip boxes to the image first.

**9. How do you handle overfitting?**
Built-in augmentation (mosaic, flips, HSV jitter), early stopping with `patience`, selecting `best.pt` on validation, and checking that train and val losses do not diverge in `results.png`.

**10. What does the confidence threshold do and how do you choose it?**
It filters low-confidence boxes at inference. Raise it for fewer false alarms, lower it for more recall. Choose it from the PR/F1 curve on validation data according to the cost of each error type (in safety, favour recall).

**11. What are the limitations of your model?**
One dataset, mostly daylight construction scenes, class imbalance between `helmet` and `head`, small/occluded heads are harder, and `head` does not distinguish caps from truly unprotected heads. I would add diverse data and per-condition evaluation.

**12. How would you take this to production?**
Export to ONNX/TensorRT, batch/stream frames, add tracking (e.g. ByteTrack) so each worker is counted once, monitor drift, log violations with snapshots, and set the threshold by business cost.
