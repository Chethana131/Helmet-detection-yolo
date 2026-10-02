"""Streamlit demo: helmet detection for images, camera snapshots and video.

Run from the repository root:
    streamlit run app/streamlit_app.py

Background: the app uses assets/background.(jpg|jpeg|png|webp) if you add one,
otherwise the bundled assets/background.svg. Keep custom images under ~1.5 MB.
"""
import base64
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.utils import (detect_image, load_model, process_video,  # noqa: E402
                       resolve_weights, to_browser_mp4)

ASSETS = config.ROOT / "assets"

st.set_page_config(page_title="Helmet Detection", page_icon="⛑️", layout="wide")


# --------------------------------------------------------------------------- #
# Background image + styling
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def background_data_uri():
    """Return the background image as a data URI (custom image first, bundled SVG as fallback)."""
    for ext, mime in (("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png"),
                      ("webp", "image/webp"), ("svg", "image/svg+xml")):
        path = ASSETS / f"background.{ext}"
        if path.exists():
            return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    return None


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --amber: #F59E0B; --yellow: #FACC15; --green: #22C55E; --red: #EF4444; --blue: #38BDF8;
        --glass: rgba(12,16,26,0.66); --line: rgba(255,255,255,0.14); }

html, body, .stApp, .stMarkdown, .stMarkdown p, .stCaption, label, button p, input, textarea,
h1, h2, h3, h4, [data-testid="stSidebar"] p, [data-baseweb="tab"] p, [data-baseweb="select"] div {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #F3F4F6; }
.stMarkdown p { font-size: 1rem; line-height: 1.6; }
[data-testid="stCaptionContainer"] { font-size: 0.9rem !important; opacity: 0.8; }

/* ---------- Background image with a readability overlay ---------- */
.stApp {
    background-image: linear-gradient(180deg, rgba(6,9,16,0.80) 0%, rgba(6,9,16,0.62) 55%, rgba(6,9,16,0.38) 100%),
                      __BG__;
    background-size: cover; background-position: center bottom; background-attachment: fixed; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1300px; }

/* ---------- Hero ---------- */
.hero { padding: 2rem 2.2rem 1.6rem 2.2rem; border-radius: 18px; margin-bottom: 1.6rem;
        background: var(--glass); border: 1px solid var(--line); backdrop-filter: blur(12px);
        box-shadow: 0 12px 44px rgba(0,0,0,0.45); }
.hero-title { font-size: 3.1rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.1;
              background: linear-gradient(90deg, #FACC15 0%, #F59E0B 55%, #FB923C 100%);
              -webkit-background-clip: text; background-clip: text;
              -webkit-text-fill-color: transparent; color: transparent; }
.hero-sub { font-size: 1.3rem; font-weight: 500; opacity: 0.88; margin-top: 0.5rem; }
.hero-desc { font-size: 1rem; line-height: 1.65; opacity: 0.72; margin-top: 0.9rem; max-width: 820px; }
.chips { margin-top: 1.2rem; display: flex; flex-wrap: wrap; gap: 0.6rem; }
.chip { font-size: 0.85rem; font-weight: 600; padding: 0.35rem 0.9rem; border-radius: 999px;
        background: rgba(245,158,11,0.16); border: 1px solid rgba(245,158,11,0.55); }
.hazard { height: 10px; border-radius: 6px; margin-top: 1.4rem;
          background: repeating-linear-gradient(45deg, #FACC15 0 14px, #111827 14px 28px); }

/* ---------- Section headings ---------- */
.section-title { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; margin: 0.6rem 0 0.2rem 0; }
.section-desc { font-size: 1rem; opacity: 0.75; margin-bottom: 1.1rem; line-height: 1.55; }
.card-title { font-size: 1.1rem; font-weight: 650; margin-bottom: 0.5rem; }

/* ---------- KPI cards ---------- */
.kpi { background: var(--glass); border: 1px solid var(--line); border-top: 4px solid var(--amber);
       border-radius: 14px; padding: 1.1rem 1.3rem 1.2rem 1.3rem; height: 100%;
       backdrop-filter: blur(12px); box-shadow: 0 8px 28px rgba(0,0,0,0.35); }
.kpi-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; opacity: 0.7; }
.kpi-value { font-size: 2.7rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15;
             margin: 0.3rem 0 0.2rem 0; font-variant-numeric: tabular-nums; }
.kpi-note { font-size: 0.85rem; opacity: 0.65; line-height: 1.4; }
.kpi-green { border-top-color: var(--green); } .kpi-red { border-top-color: var(--red); }
.kpi-blue { border-top-color: var(--blue); }

/* ---------- Result banners ---------- */
.banner { border-radius: 12px; padding: 1.05rem 1.3rem; margin: 1.1rem 0; font-size: 1.08rem; line-height: 1.5;
          border: 1px solid var(--line); border-left-width: 7px; backdrop-filter: blur(10px); }
.banner-warn { border-left-color: var(--red); background: rgba(239,68,68,0.20); }
.banner-ok { border-left-color: var(--green); background: rgba(34,197,94,0.18); }

/* ---------- Bordered containers (frosted glass) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(12,16,26,0.58); backdrop-filter: blur(10px);
        border-radius: 14px; border-color: var(--line) !important; }

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] { padding: 0.7rem 1.2rem; }
button[data-baseweb="tab"] p { font-size: 1.05rem; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--amber); }
div[data-baseweb="tab-highlight"] { background-color: var(--amber) !important; height: 3px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] > div { background: rgba(10,14,22,0.88); backdrop-filter: blur(12px); }
section[data-testid="stSidebar"] h2 { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }
section[data-testid="stSidebar"] label p { font-size: 0.95rem; font-weight: 600; }
.side-card { border: 1px solid var(--line); border-left: 4px solid var(--amber); border-radius: 10px;
             padding: 0.8rem 1rem; background: rgba(255,255,255,0.05); margin: 0.9rem 0; }
.side-row { display: flex; justify-content: space-between; gap: 0.8rem; font-size: 0.88rem; padding: 0.2rem 0; }
.side-row span { opacity: 0.7; } .side-row b { text-align: right; font-weight: 650; }
.side-help { font-size: 0.85rem; line-height: 1.55; opacity: 0.8; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
</style>
"""
uri = background_data_uri()
st.markdown(CSS.replace("__BG__", f'url("{uri}")' if uri else "none"), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def render_header():
    chips = "".join(f'<span class="chip">{c}</span>' for c in
                    ["YOLOv8", "OpenCV", "Streamlit", "helmet · head · person"])
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">Helmet Detection</div>
          <div class="hero-sub">Real-time PPE safety monitoring with YOLOv8</div>
          <div class="hero-desc">Upload a photo, take a snapshot or run a video. The model finds every worker,
          checks whether they wear a helmet, and flags bare heads as safety violations.</div>
          <div class="chips">{chips}</div>
          <div class="hazard"></div>
        </div>
        """,
        unsafe_allow_html=True)


def section(title, description=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="section-desc">{description}</div>', unsafe_allow_html=True)


def kpi_card(label, value, note="", tone=""):
    """Big-number card. tone: '' (amber), 'green', 'red' or 'blue'."""
    cls = f"kpi kpi-{tone}" if tone else "kpi"
    st.markdown(f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                f'<div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


def banner(kind, title, text):
    st.markdown(f'<div class="banner banner-{kind}"><b>{title}</b> {text}</div>', unsafe_allow_html=True)


def render_counts(counts):
    """KPI cards + safety banner from a Counter of class names."""
    helmets, bare = counts.get("helmet", 0), counts.get(config.VIOLATION_CLASS, 0)
    persons, total = counts.get("person", 0), sum(counts.values())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Helmets", helmets, "Workers wearing a helmet", "green")
    with c2:
        kpi_card("Bare heads", bare, "Safety violations", "red" if bare else "green")
    with c3:
        kpi_card("Persons", persons, "Persons detected", "blue")
    with c4:
        kpi_card("Detections", total, "All boxes above the threshold")
    if bare:
        banner("warn", "Safety violation.", f"{bare} worker(s) detected WITHOUT a helmet ({helmets} with helmet).")
    else:
        banner("ok", "All clear.", f"{helmets} helmet(s) detected and no bare heads.")


@st.cache_resource(show_spinner="Loading model...")
def get_model(weights):
    return load_model(weights)


# --------------------------------------------------------------------------- #
# Header + sidebar
# --------------------------------------------------------------------------- #
render_header()

with st.sidebar:
    st.markdown("## Controls")
    weights, is_custom = resolve_weights(st.text_input("Weights path (optional)", ""))
    conf = st.slider("Confidence threshold", 0.05, 0.95, config.DEFAULT_CONF, 0.05,
                     help="Detections with lower confidence are discarded.")
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, config.DEFAULT_IOU, 0.05,
                    help="Overlapping boxes above this IoU are merged into one.")
    dot = "var(--green)" if is_custom else "var(--red)"
    st.markdown(
        f"""
        <div class="side-card">
          <div class="side-row"><span>Weights</span><b><span class="dot" style="background:{dot}"></span>{Path(weights).name}</b></div>
          <div class="side-row"><span>Confidence</span><b>{conf:.2f}</b></div>
          <div class="side-row"><span>NMS IoU</span><b>{iou:.2f}</b></div>
        </div>
        <div class="side-help"><b>Confidence:</b> higher = fewer false alarms but more missed objects.<br>
        <b>NMS IoU:</b> how much two boxes may overlap before they are merged into one.</div>
        """,
        unsafe_allow_html=True)

if not is_custom:
    st.warning("No trained weights found at `models/best.pt`. Using generic COCO weights, which do NOT "
               "know the helmet classes. Train the model first (see README).")
model = get_model(weights)

tab_img, tab_vid, tab_perf, tab_about = st.tabs(
    ["Image detection", "Video detection", "Model performance", "How it works"])


# --------------------------------------------------------------------------- #
# Tab 1: images / camera
# --------------------------------------------------------------------------- #
def render_image_tab():
    section("Image detection", "Upload a photo or take a snapshot. Boxes, classes and confidences are drawn on the result.")
    mode = st.radio("Input", ["Upload an image", "Use the camera"], horizontal=True, label_visibility="collapsed")
    if mode == "Upload an image":
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="img")
    else:
        up = st.camera_input("Take a photo")
    if up is None:
        st.info("Add an image to see detections and the safety summary.")
        return

    img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("That file could not be read as an image.")
        return
    annotated, counts, result = detect_image(model, img, conf, iou)

    render_counts(counts)
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="card-title">Original</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown('<div class="card-title">Detections</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
            ok, buf = cv2.imencode(".png", annotated)
            if ok:
                st.download_button("Download annotated image", buf.tobytes(), "helmet_detection.png", "image/png")

    if len(result.boxes):
        st.write("")
        section("Detected objects")
        rows = [{"class": model.names[int(c)], "confidence": round(float(s), 3)}
                for c, s in zip(result.boxes.cls, result.boxes.conf)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "class": st.column_config.TextColumn("Class"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.3f")})


# --------------------------------------------------------------------------- #
# Tab 2: video
# --------------------------------------------------------------------------- #
def render_video_tab():
    section("Video detection", "Every frame is processed. Frames that contain a bare head are counted as violations.")
    upv = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if upv is None:
        st.info("Upload a short video (a few seconds is plenty) to run detection on every frame.")
        return

    if st.button("Run detection on video", type="primary"):
        tmp = Path(tempfile.mkdtemp())
        src = tmp / upv.name
        src.write_bytes(upv.getvalue())
        raw_out, web_out = tmp / "out_raw.mp4", tmp / "out_web.mp4"
        bar = st.progress(0.0, text="Processing frames...")
        stats = process_video(model, src, raw_out, conf, iou, progress_cb=lambda f: bar.progress(f))
        bar.empty()
        playable = web_out if to_browser_mp4(raw_out, web_out) else raw_out
        st.session_state["video_result"] = {"name": upv.name, "bytes": playable.read_bytes(), "stats": stats}

    res = st.session_state.get("video_result")
    if res and res["name"] == upv.name:
        s = res["stats"]
        pct = s["frames_with_violation"] / s["frames"] if s["frames"] else 0.0
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Frames processed", f"{s['frames']:,}", f"Video at {s['fps']:.0f} fps")
        with c2:
            kpi_card("Frames with a violation", f"{s['frames_with_violation']:,}", "At least one bare head",
                     "red" if s["frames_with_violation"] else "green")
        with c3:
            kpi_card("Violation rate", f"{pct:.1%}", "Share of frames", "red" if pct else "green")
        if s["frames_with_violation"]:
            banner("warn", "Safety violations found.", "Some frames contain a worker without a helmet.")
        else:
            banner("ok", "All clear.", "No bare heads were detected in this video.")
        with st.container(border=True):
            st.markdown('<div class="card-title">Annotated video</div>', unsafe_allow_html=True)
            st.video(res["bytes"])


# --------------------------------------------------------------------------- #
# Tab 3: model performance (real numbers from results/metrics.json)
# --------------------------------------------------------------------------- #
def first_existing(folder, names):
    return next((folder / n for n in names if (folder / n).exists()), None)


def render_performance_tab():
    path = config.RESULTS_DIR / "metrics.json"
    if not path.exists():
        st.info("No evaluation results yet. Run `python -m src.evaluate` to generate them.")
        return
    data = json.loads(path.read_text())
    o, split = data["overall"], data.get("split", "test")
    section("Model performance", f"Evaluated on the held-out {split} split, which was never used for training.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Precision", f"{o['precision']:.3f}", "Detections that were correct")
    with c2:
        kpi_card("Recall", f"{o['recall']:.3f}", "Real objects that were found", "green")
    with c3:
        kpi_card("mAP@0.5", f"{o['mAP@0.5']:.3f}", "Boxes matching at IoU 0.5", "blue")
    with c4:
        kpi_card("mAP@0.5:0.95", f"{o['mAP@0.5:0.95']:.3f}", "Stricter, averaged over IoU 0.5-0.95")

    st.write("")
    section("Per-class results")
    rows = [{"class": k, **v} for k, v in data["per_class"].items()]
    bar = dict(min_value=0.0, max_value=1.0, format="%.3f")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
        "class": st.column_config.TextColumn("Class"),
        "precision": st.column_config.ProgressColumn("Precision", **bar),
        "recall": st.column_config.ProgressColumn("Recall", **bar),
        "f1": st.column_config.ProgressColumn("F1", **bar),
        "mAP@0.5": st.column_config.ProgressColumn("mAP@0.5", **bar),
        "mAP@0.5:0.95": st.column_config.ProgressColumn("mAP@0.5:0.95", **bar)})

    folder = config.RESULTS_DIR / f"{split}_eval"
    pr = first_existing(folder, ["BoxPR_curve.png", "PR_curve.png"])
    cm = first_existing(folder, ["confusion_matrix_normalized.png", "confusion_matrix.png"])
    if pr or cm:
        st.write("")
        section("Evaluation plots")
        g1, g2 = st.columns(2)
        for col, img, title in ((g1, pr, "Precision-Recall curve"), (g2, cm, "Confusion matrix")):
            if img:
                with col, st.container(border=True):
                    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
                    st.image(str(img), width="stretch")


# --------------------------------------------------------------------------- #
# Tab 4: how it works
# --------------------------------------------------------------------------- #
def render_about_tab():
    section("How it works", "A single neural network looks at the image once and predicts every box and class.")
    a, b, c = st.columns(3)
    for col, title, text in (
        (a, "1. Detect", "YOLOv8, fine-tuned on 5,000 construction-site images, predicts a box, a class and a "
                         "confidence score for every object."),
        (b, "2. Filter", "Boxes below the **confidence threshold** are dropped. **Non-maximum suppression** merges "
                         "overlapping boxes using the **IoU** threshold."),
        (c, "3. Flag", "A `head` detection means a bare head, which is counted as a safety violation and "
                       "highlighted in the summary.")):
        with col, st.container(border=True):
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(text)
    st.write("")
    with st.container(border=True):
        st.markdown('<div class="card-title">Classes</div>', unsafe_allow_html=True)
        st.markdown("- **helmet**: a worker wearing a helmet\n"
                    "- **head**: a bare head, i.e. a worker **without** a helmet (violation)\n"
                    "- **person**: a person")
        st.caption("Metrics, plots and the training code are in the repository README.")


with tab_img:
    render_image_tab()
with tab_vid:
    render_video_tab()
with tab_perf:
    render_performance_tab()
with tab_about:
    render_about_tab()

st.divider()
st.caption("Built with Ultralytics YOLOv8, OpenCV and Streamlit. Dataset: Kaggle Hard Hat Detection.")
