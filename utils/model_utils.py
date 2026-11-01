"""Wrapper around the fine-tuned micro-SAM model (vit_b_lm, quadrant split).

run_segmentation() keeps the same interface the pages already use.
If the checkpoint file is missing, it falls back to the placeholder mode
so the rest of the app still works.
"""
from pathlib import Path
import numpy as np

MODEL_TYPE = "vit_b_lm"
# models/ folder in the project root, so it works wherever you run streamlit from
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "microsam_platelets_ft_quadrant.pt"

_model_cache = {}


def _load_model():
    if "model" not in _model_cache:
        from micro_sam.util import get_device
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=MODEL_TYPE, checkpoint=str(MODEL_PATH),
            device=get_device(), segmentation_mode="ais")
        _model_cache["model"] = (predictor, segmenter)
    return _model_cache["model"]


def compute_scaling(stack):
    """Scaling limits from the whole stack, the same way the model was trained."""
    lo, hi = np.percentile(stack, [1, 99.8])
    return float(lo), float(hi)


def to_uint8(frame, scaling=None):
    """Scale a frame to 0-255. Pass the stack's scaling to match training."""
    if frame.dtype == np.uint8 and scaling is None:
        return frame
    if scaling is None:
        scaling = compute_scaling(frame)   # fallback: per-frame scaling
    lo, hi = scaling
    out = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def run_segmentation(frame, placeholder_mask=None, scaling=None):
    """Run segmentation on a single 2D frame.

    If the micro-SAM checkpoint exists, runs the fine-tuned model.
    Otherwise, returns `placeholder_mask` if supplied (e.g. the GT mask)
    so the rest of the app remains usable.
    """
    if model_is_ready():
        from micro_sam.automatic_segmentation import automatic_instance_segmentation
        predictor, segmenter = _load_model()
        seg = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter,
            input_path=to_uint8(frame, scaling), ndim=2, verbose=False)
        return seg.astype(np.int32)

    if placeholder_mask is not None:
        return placeholder_mask

    return np.zeros_like(frame, dtype=np.int32)


def segment_stack(stack, progress_callback=None):
    """Segment every frame in a (T, H, W) stack, using whole-stack scaling."""
    scaling = compute_scaling(stack)
    masks = np.zeros(stack.shape, dtype=np.int32)
    for t in range(stack.shape[0]):
        masks[t] = run_segmentation(stack[t], scaling=scaling)
        if progress_callback is not None:
            progress_callback(t + 1, stack.shape[0])
    return masks


def model_is_ready():
    return MODEL_PATH.exists()