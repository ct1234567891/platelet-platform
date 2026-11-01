"""IO helpers for loading platelet TIRF image stacks and masks."""
import numpy as np
from tifffile import imread


def load_image_stack(path):
    """Load a multi-frame .tif image stack. Returns array of shape (frames, H, W)."""
    return imread(path)


def load_mask_stack(path):
    """Load a mask stack. Supports .tif (array) or .npz (dict-like archive).

    Returns array of shape (frames, H, W) with integer labels.
    """
    if path.lower().endswith(".npz"):
        data = np.load(path)
        # try common key names used across the lab's pipelines
        for key in ("arr_0", "masks", "mask"):
            if key in data:
                return data[key]
        # fallback: first array in the archive
        first_key = list(data.keys())[0]
        return data[first_key]
    return imread(path)


def validate_stack_pair(images, masks):
    """Basic sanity checks that an image stack and mask stack are compatible.

    Returns a list of human-readable warning strings (empty list if all good).
    """
    errors = []
    if images.shape[0] != masks.shape[0]:
        errors.append(
            f"Frame count mismatch: images has {images.shape[0]} frames, "
            f"masks has {masks.shape[0]} frames."
        )
    if images.shape[-2:] != masks.shape[-2:]:
        errors.append(
            f"Spatial size mismatch: images {images.shape[-2:]}, masks {masks.shape[-2:]}."
        )
    return errors
