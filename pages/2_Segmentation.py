"""Segmentation -- run micro-SAM on the chosen frame and compare to GT."""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from skimage.color import label2rgb

from utils.model_utils import run_segmentation, model_is_ready, compute_scaling, to_uint8, MODEL_PATH

st.set_page_config(page_title="Segmentation", layout="wide")
st.title("Segmentation")

if not model_is_ready():
    st.warning(
        f"Model file not found at {MODEL_PATH}. Running in placeholder mode: "
        "the loaded ground-truth mask is shown as a stand-in prediction."
    )


def count_cells(mask):
    """Number of platelets in a mask (IDs don't have to be consecutive)."""
    return int((np.unique(mask) > 0).sum())


if "images" not in st.session_state:
    st.info("Go to 'Upload & View' first and load an image stack.")
else:
    images = st.session_state["images"]
    n_frames = images.shape[0]

    # whole-stack scaling, same as in training; worked out once per loaded stack
    if st.session_state.get("scaling_for") != id(images):
        st.session_state["scaling"] = compute_scaling(images)
        st.session_state["scaling_for"] = id(images)
    scaling = st.session_state["scaling"]

    frame_idx = st.slider("Frame", 0, n_frames - 1, 0, key="seg_frame")
    gt_mask = st.session_state["masks"][frame_idx] if "masks" in st.session_state else None

    if st.button("Run segmentation on this frame"):
        with st.spinner("Segmenting... (the first run also loads the model)"):
            st.session_state["last_pred_mask"] = run_segmentation(
                images[frame_idx], placeholder_mask=gt_mask, scaling=scaling)
        st.session_state["last_pred_frame_idx"] = frame_idx

    if "last_pred_mask" in st.session_state and st.session_state.get("last_pred_frame_idx") == frame_idx:
        pred_mask = st.session_state["last_pred_mask"]
        frame8 = to_uint8(images[frame_idx], scaling)

        panels = [("Original", None)]
        if gt_mask is not None:
            panels.append((f"Ground truth ({count_cells(gt_mask)} platelets)", gt_mask))
        panels.append((f"Predicted ({count_cells(pred_mask)} platelets)", pred_mask))

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, (title, lab) in zip(axes, panels):
            if lab is None:
                ax.imshow(frame8, cmap="gray")
            else:
                ax.imshow(label2rgb(lab, image=frame8, bg_label=0, alpha=0.4))
            ax.set_title(title)
            ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)