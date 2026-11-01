"""Metrics -- segment the whole video once, then score it against the ground truth."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.model_utils import segment_stack, model_is_ready
from utils.metrics_utils import match_instances

st.set_page_config(page_title="Metrics", layout="wide")
st.title("Metrics")
st.caption(
    "Benchmarks (Sellier, 2025): Dice > 0.90, Average Precision > 0.80 @ IoU 0.50, "
    "Tracking Accuracy > 0.90"
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TEST_REGION = "Test quadrant only (q2_tr)"


def stack_id(images):
    """Short fingerprint of the loaded video, used to name its saved predictions."""
    step = max(1, images.shape[0] // 10)
    return hashlib.md5(np.ascontiguousarray(images[::step]).tobytes()).hexdigest()[:10]


def crop_region(mask, region):
    if region == TEST_REGION:
        h, w = mask.shape
        return mask[: h // 2, w // 2:]
    return mask


if "images" not in st.session_state or "masks" not in st.session_state:
    st.info("Load both an image stack and a mask stack on 'Upload & View' first.")
elif not model_is_ready():
    st.warning("Model file not found in models/, so there's nothing to evaluate yet.")
else:
    images = st.session_state["images"]
    masks = st.session_state["masks"]
    n_frames = images.shape[0]
    pred_path = RESULTS_DIR / f"pred_masks_{stack_id(images)}.npz"

    # ---- 1. Predictions for the whole video: load saved ones, or run the model once ----
    if st.session_state.get("pred_for") != pred_path.name and pred_path.exists():
        st.session_state["pred_masks"] = np.load(pred_path)["masks"]
        st.session_state["pred_for"] = pred_path.name

    if st.session_state.get("pred_for") != pred_path.name:
        minutes = max(1, round(n_frames * 1.8 / 60))
        st.write(
            f"No saved predictions for this video yet. Segmenting all {n_frames} frames "
            f"takes about {minutes} minutes and only needs doing once."
        )
        if st.button("Segment all frames"):
            bar = st.progress(0.0, text="Starting...")

            def update(done, total):
                bar.progress(done / total, text=f"Segmenting frame {done} of {total}")

            pred = segment_stack(images, progress_callback=update)
            RESULTS_DIR.mkdir(exist_ok=True)
            np.savez_compressed(pred_path, masks=pred.astype(np.uint16))
            st.session_state["pred_masks"] = pred
            st.session_state["pred_for"] = pred_path.name
            st.rerun()
        st.stop()

    pred_masks = st.session_state["pred_masks"]
    st.success(f"Using saved predictions for all {n_frames} frames ({pred_path.name}).")

    # ---- 2. Choose what to score ----
    region = st.radio("Region to score", ["All quadrants", TEST_REGION], horizontal=True)
    if region == "All quadrants":
        st.caption(
            "This includes the three quadrants the model was trained on, so scores will be "
            "higher than a fair test. The test-quadrant option is the fairer view."
        )
    else:
        st.caption(
            "Scored on the top-right quadrant, which the model never saw in training. "
            "Numbers will be close to, but not exactly the same as, the notebook results, "
            "because here the model runs on full frames rather than 256x256 crops."
        )
    start, end = st.slider("Frames", 0, n_frames - 1, (0, n_frames - 1))

    # ---- 3. Score every frame with the same matching as the notebooks ----
    rows, all_ious = [], []
    for t in range(start, end + 1):
        tp, fp, fn, ious = match_instances(crop_region(masks[t], region),
                                           crop_region(pred_masks[t], region))
        all_ious += ious
        denom = tp + fp + fn
        rows.append({"frame": t, "real": tp + fn, "predicted": tp + fp,
                     "tp": tp, "fp": fp, "fn": fn,
                     "AP@0.5": tp / denom if denom else np.nan})
    df = pd.DataFrame(rows)

    TP, FP, FN = int(df.tp.sum()), int(df.fp.sum()), int(df.fn.sum())
    all_ious = np.array(all_ious)
    ap = TP / max(TP + FP + FN, 1)
    dice = float(np.mean(2 * all_ious / (1 + all_ious))) if len(all_ious) else 0.0
    iou = float(all_ious.mean()) if len(all_ious) else 0.0

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AP@0.5", f"{ap:.4f}", f"{ap - 0.80:+.4f} vs 0.80")
    c2.metric("Matched Dice", f"{dice:.4f}", f"{dice - 0.90:+.4f} vs 0.90")
    c3.metric("Matched IoU", f"{iou:.4f}")
    c4.metric("TP / FP / FN", f"{TP} / {FP} / {FN}")

    st.subheader("Per frame")
    st.line_chart(df.set_index("frame")[["AP@0.5"]])
    st.line_chart(df.set_index("frame")[["fp", "fn"]])
    st.dataframe(df, use_container_width=True)

    name = "test_quadrant" if region == TEST_REGION else "all_quadrants"
    st.download_button("Download per-frame results (CSV)", df.to_csv(index=False),
                       file_name=f"microsam_metrics_{name}_frames_{start}-{end}.csv")