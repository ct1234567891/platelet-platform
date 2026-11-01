"""Upload & View -- load an image/mask stack and step through frames with mask overlay."""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from utils.io_utils import load_image_stack, load_mask_stack, validate_stack_pair

st.set_page_config(page_title="Upload & View", layout="wide")
st.title("Upload & View")

col1, col2 = st.columns(2)
with col1:
    image_file = st.file_uploader("Image stack (.tif)", type=["tif", "tiff"])
with col2:
    mask_file = st.file_uploader("Mask stack (.npz or .tif)", type=["npz", "tif", "tiff"])


def _save_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


if image_file is not None:
    image_path = _save_temp(image_file)
    st.session_state["images"] = load_image_stack(image_path)
    st.session_state["image_path"] = image_file.name

if mask_file is not None:
    mask_path = _save_temp(mask_file)
    st.session_state["masks"] = load_mask_stack(mask_path)
    st.session_state["mask_path"] = mask_file.name

if "images" in st.session_state:
    images = st.session_state["images"]
    n_frames = images.shape[0]
    st.success(
        f"Loaded {st.session_state['image_path']} -- {n_frames} frames, "
        f"{images.shape[-2]}x{images.shape[-1]} px"
    )

    if "masks" in st.session_state:
        for e in validate_stack_pair(images, st.session_state["masks"]):
            st.warning(e)

    frame_idx = st.slider("Frame", 0, n_frames - 1, 0)
    show_mask = st.checkbox("Show mask overlay", value="masks" in st.session_state)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(images[frame_idx], cmap="gray")
    if show_mask and "masks" in st.session_state:
        masks = st.session_state["masks"]
        mask_frame = np.ma.masked_where(masks[frame_idx] == 0, masks[frame_idx])
        ax.imshow(mask_frame, cmap="tab20b", alpha=0.5)
        st.caption(f"Frame {frame_idx}: {int(masks[frame_idx].max())} labelled cells")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("Upload a .tif image stack to get started.")
