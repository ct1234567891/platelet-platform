"""Segmentation evaluation metrics: aggregate and per-cell Dice / IoU.

Aggregate metrics treat the whole frame as one blob (fast, but not what the
Sellier (2025) AP@IoU0.50 benchmark actually measures). per_cell_metrics()
matches predicted cells to ground-truth cells using the Hungarian algorithm on
pixel overlap, then reports IoU/Dice per matched pair -- closer to true
instance-segmentation evaluation.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label


def dice_coef(mask_true, mask_pred):
    intersection = np.sum((mask_true > 0) & (mask_pred > 0))
    size_sum = np.sum(mask_true > 0) + np.sum(mask_pred > 0)
    if size_sum == 0:
        return 1.0
    return 2.0 * intersection / size_sum


def iou_coef(mask_true, mask_pred):
    intersection = np.sum((mask_true > 0) & (mask_pred > 0))
    union = np.sum((mask_true > 0) | (mask_pred > 0))
    if union == 0:
        return 1.0
    return intersection / union


def per_cell_metrics(mask_true, mask_pred):
    """Match predicted cells to GT cells (Hungarian algorithm on overlap) and
    return (ious, dices) lists -- one entry per matched pair.
    """
    labels_true = label(mask_true) if mask_true.max() <= 1 else mask_true
    labels_pred = label(mask_pred) if mask_pred.max() <= 1 else mask_pred

    num_true = int(labels_true.max())
    num_pred = int(labels_pred.max())
    if num_true == 0 or num_pred == 0:
        return [], []

    overlap = np.zeros((num_true, num_pred), dtype=np.int64)
    for i in range(1, num_true + 1):
        true_inst = labels_true == i
        for j in range(1, num_pred + 1):
            pred_inst = labels_pred == j
            overlap[i - 1, j - 1] = np.logical_and(true_inst, pred_inst).sum()

    row_ind, col_ind = linear_sum_assignment(-overlap)

    ious, dices = [], []
    for i, j in zip(row_ind, col_ind):
        intersection = overlap[i, j]
        if intersection == 0:
            continue
        area_true = int((labels_true == i + 1).sum())
        area_pred = int((labels_pred == j + 1).sum())
        union = area_true + area_pred - intersection
        ious.append(intersection / union)
        dices.append((2 * intersection) / (area_true + area_pred))
    return ious, dices



# --- Instance matching, the same calculation as the training notebooks ---
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_instances(gt, pr, thr=0.5):
    """One-to-one matching of platelets (Hungarian algorithm on IoU).

    Returns (tp, fp, fn, list of IoUs for matched pairs). A match counts at IoU >= thr.
    """
    gt = gt.astype(np.int64); pr = pr.astype(np.int64)
    n_pr = pr.max() + 1
    overlap = np.bincount((gt * n_pr + pr).ravel(),
                          minlength=(gt.max() + 1) * n_pr).reshape(-1, n_pr)
    union = overlap.sum(1, keepdims=True) + overlap.sum(0, keepdims=True) - overlap
    iou = overlap / np.maximum(union, 1)
    gt_ids = np.unique(gt); gt_ids = gt_ids[gt_ids > 0]
    pr_ids = np.unique(pr); pr_ids = pr_ids[pr_ids > 0]
    iou = iou[np.ix_(gt_ids, pr_ids)]
    if iou.size == 0:
        return 0, len(pr_ids), len(gt_ids), []
    r, c = linear_sum_assignment(-iou)
    matched = iou[r, c]
    matched = matched[matched >= thr]
    tp = len(matched)
    return tp, len(pr_ids) - tp, len(gt_ids) - tp, list(matched)