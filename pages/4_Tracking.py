"""Tracking -- link predicted platelets across frames so each keeps one ID."""
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage

from utils.model_utils import compute_scaling, to_uint8
from utils.tracking_utils import link_frames, track_table, link_accuracy, tracking_scores, Q2_TR

st.set_page_config(page_title="Tracking", layout="wide")
st.title("Tracking")
st.caption(
    "Each predicted platelet is linked to the platelet it overlaps most in the "
    "previous frames, so it keeps the same ID through the video."
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def stack_id(images):
    """Same fingerprint as the Metrics page, to find the saved predictions."""
    step = max(1, images.shape[0] // 10)
    return hashlib.md5(np.ascontiguousarray(images[::step]).tobytes()).hexdigest()[:10]


if "images" not in st.session_state:
    st.info("Go to 'Upload & View' first and load an image stack.")
    st.stop()

images = st.session_state["images"]
n_frames = images.shape[0]
pred_path = RESULTS_DIR / f"pred_masks_{stack_id(images)}.npz"

# ---- 1. Saved predictions from the Metrics page ----
if st.session_state.get("pred_for") != pred_path.name:
    if pred_path.exists():
        st.session_state["pred_masks"] = np.load(pred_path)["masks"]
        st.session_state["pred_for"] = pred_path.name
    else:
        st.info("No saved predictions for this video yet. Go to 'Metrics' and click "
                "'Segment all frames' first.")
        st.stop()
pred_masks = st.session_state["pred_masks"]

# ---- 2. Link platelets across frames ----
c1, c2 = st.columns(2)
min_iou = c1.slider("Minimum overlap to link (IoU)", 0.1, 0.9, 0.3, 0.05)
max_gap = c2.slider("Frames a platelet can be missed and still keep its ID", 0, 5, 2)

key = (pred_path.name, min_iou, max_gap)
if st.session_state.get("tracked_for") != key or "tra" not in st.session_state:
    with st.spinner("Linking platelets and scoring tracks..."):
        tracked = link_frames(pred_masks, min_iou=min_iou, max_gap=max_gap)
        st.session_state["tracked"] = tracked
        st.session_state["tracks"] = pd.DataFrame(track_table(tracked))
        st.session_state["tra"] = None
        if "masks" in st.session_state:
            gt = st.session_state["masks"]
            st.session_state["link_acc"] = link_accuracy(gt[:, Q2_TR[0], Q2_TR[1]],
                                                         tracked[:, Q2_TR[0], Q2_TR[1]])
            st.session_state["tra"] = tracking_scores(gt, tracked)
    st.session_state["tracked_for"] = key
tracked = st.session_state["tracked"]
tracks = st.session_state["tracks"]
if tracks.empty:
    st.warning("No platelets were found in the saved predictions, so there is nothing to track.")
    st.stop()

# ---- 3. Summary ----
# ---- 3. Summary ----
st.subheader("Summary")
cols = st.columns(3)
cols[0].metric("Tracks found", len(tracks))
cols[1].metric("Tracks through whole video", int((tracks.frames == n_frames).sum()))
cols[2].metric("Median track length (frames)", int(tracks.frames.median()))

if "masks" in st.session_state and st.session_state.get("tra") is not None:
    acc, correct, total = st.session_state["link_acc"]
    scores = st.session_state["tra"]
    gt_q = st.session_state["masks"][:, Q2_TR[0], Q2_TR[1]]
    n_real = int((np.unique(gt_q) > 0).sum())

    st.subheader("Tracking quality (test quadrant)")
    q = st.columns(3)
    if scores.get("TRA") is None:
        q[0].metric("Tracking score (TRA)", "—")
        st.warning("The track files didn't pass the format check, so TRA couldn't be calculated.")
    else:
        q[0].metric("Tracking score (TRA)", f"{scores['TRA']:.4f}")
    q[1].metric("Link accuracy", f"{acc:.4f}", f"{acc - 0.90:+.4f} vs 0.90")
    q[2].metric("ID switches", total - correct)
    st.caption(
        f"Scored on the upper-right quadrant (q2_tr), which the model never saw in training. "
        f"The ground truth has {n_real} real platelets there. TRA is the Cell Tracking "
        "Challenge tracking score. It counts how many fixes (adding, removing or relinking "
        "platelets) it would take to turn the predicted tracks into the ground truth, so it "
        "penalises missed and extra platelets as well as wrong links. Link accuracy is a "
        f"simpler check: {correct} of {total} links kept the same ID as the last time that "
        "real platelet was detected."
    )

# ---- 4. Viewer and track table ----
if st.session_state.get("scaling_for") != id(images):
    st.session_state["scaling"] = compute_scaling(images)
    st.session_state["scaling_for"] = id(images)
scaling = st.session_state["scaling"]

left, right = st.columns([3, 2])
with left:
    st.subheader("View tracks")
    t = st.slider("Frame", 0, n_frames - 1, 0, key="track_frame")
    show_ids = st.checkbox("Show track IDs", value=True)

    lab = tracked[t]
    ids = np.unique(lab); ids = ids[ids > 0]
    colors = np.random.default_rng(0).uniform(0.2, 1.0, (int(tracked.max()) + 1, 3))
    gray = np.repeat((to_uint8(images[t], scaling) / 255.0)[..., None], 3, axis=2)
    overlay = np.where(lab[..., None] > 0, 0.55 * colors[lab] + 0.45 * gray, gray)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay)
    if show_ids and len(ids):
        for tid, (y, x) in zip(ids, ndimage.center_of_mass(lab > 0, lab, ids)):
            ax.text(x, y, str(tid), color="white", fontsize=7, ha="center", va="center")
    ax.set_title(f"Frame {t}: {len(ids)} platelets")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Tracks")
    st.dataframe(tracks.sort_values("frames", ascending=False),
                 width="stretch", hide_index=True)
    st.download_button("Download tracks (CSV)", tracks.to_csv(index=False),
                       file_name="microsam_tracks.csv")
