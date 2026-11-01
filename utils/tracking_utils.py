
import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(a, b):
    """IoU between every object in label image a and every object in b."""
    a = a.astype(np.int64); b = b.astype(np.int64)
    nb = b.max() + 1
    overlap = np.bincount((a * nb + b).ravel(), minlength=(a.max() + 1) * nb).reshape(-1, nb)
    union = overlap.sum(1, keepdims=True) + overlap.sum(0, keepdims=True) - overlap
    iou = overlap / np.maximum(union, 1)
    a_ids = np.unique(a); a_ids = a_ids[a_ids > 0]
    b_ids = np.unique(b); b_ids = b_ids[b_ids > 0]
    return iou[np.ix_(a_ids, b_ids)], a_ids, b_ids


def _match(a, b, min_iou):
    """Dict {id in a: id in b} for one-to-one matches with IoU >= min_iou."""
    iou, a_ids, b_ids = _iou_matrix(a, b)
    if iou.size == 0:
        return {}
    r, c = linear_sum_assignment(-iou)
    return {int(a_ids[i]): int(b_ids[j]) for i, j in zip(r, c) if iou[i, j] >= min_iou}


def _pixels_by_id(labels):
    """Dict {id: flat pixel indices} for every object in a label image."""
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    ids, starts = np.unique(flat[order], return_index=True)
    groups = np.split(order, starts[1:])
    return {int(i): g for i, g in zip(ids, groups) if i > 0}


def link_frames(masks, min_iou=0.3, max_gap=2):
    """Give each platelet the same ID in every frame it appears in.

    masks: (T, H, W) predicted masks. Returns tracked masks of the same shape.
    min_iou: how much a platelet must overlap its previous position to be linked.
    max_gap: a platelet missed for up to this many frames can still be linked
             back to its old ID when it reappears.
    """
    tracked = np.zeros(masks.shape, dtype=np.int32)
    next_id = 1
    last_seen = {}   # track id -> (frame it was last seen, its pixels then)
    for t in range(masks.shape[0]):
        cur = masks[t]
        cur_ids = np.unique(cur); cur_ids = cur_ids[cur_ids > 0]

        # where each recent track was last seen (previous frame, plus short gaps)
        memory = np.zeros(cur.size, dtype=np.int64)
        for tid, (seen_t, pix) in sorted(last_seen.items(), key=lambda kv: kv[1][0]):
            if t - seen_t <= max_gap + 1:
                memory[pix] = tid
        memory = memory.reshape(cur.shape)

        links = _match(cur, memory, min_iou) if memory.any() else {}
        lut = np.zeros(int(cur.max()) + 1, dtype=np.int32)
        for cid in cur_ids:
            if int(cid) in links:
                lut[cid] = links[int(cid)]
            else:
                lut[cid] = next_id
                next_id += 1
        tracked[t] = lut[cur]

        for tid, pix in _pixels_by_id(tracked[t]).items():
            last_seen[tid] = (t, pix)
    return tracked


def track_table(tracked):
    """One row per track: first and last frame, number of frames, mean area in pixels."""
    info = {}
    for t in range(tracked.shape[0]):
        areas = np.bincount(tracked[t].ravel())
        for tid in np.nonzero(areas)[0]:
            if tid == 0:
                continue
            row = info.setdefault(int(tid), {"track_id": int(tid), "first_frame": t,
                                             "last_frame": t, "frames": 0, "area_sum": 0})
            row["last_frame"] = t
            row["frames"] += 1
            row["area_sum"] += int(areas[tid])
    rows = []
    for row in info.values():
        row["mean_area_px"] = round(row.pop("area_sum") / row["frames"], 1)
        rows.append(row)
    return rows


def link_accuracy(gt_masks, tracked, match_iou=0.5):
    """How often the tracker keeps the same ID for the same real platelet.

    Each time a real platelet is detected, its track ID is compared with the
    track ID it had the last time it was detected. Returns
    (accuracy, correct links, total links). This is a simple consistency check,
    not the formal Cell Tracking Challenge TRA metric.
    """
    correct = total = 0
    last_track = {}   # real id -> track id when last detected
    for t in range(gt_masks.shape[0]):
        for gid, tid in _match(gt_masks[t], tracked[t], match_iou).items():
            if gid in last_track:
                total += 1
                correct += int(last_track[gid] == tid)
            last_track[gid] = tid
    return (correct / total if total else float("nan")), correct, total


# ---- Cell Tracking Challenge TRA ----
import os
import tempfile
import tifffile

Q2_TR = (slice(0, 256), slice(256, 512))   # upper-right test quadrant


def _split_gaps(masks):
    """Relabel so each label sits in one unbroken run of frames.

    The Cell Tracking Challenge format doesn't allow gaps inside a track, so a
    platelet that goes missing and comes back is split into a new label whose
    parent is the earlier piece. Returns (relabelled masks, [label, first, last, parent] rows).
    """
    ids_per_frame = [np.unique(m) for m in masks]
    frames_by_id = {}
    for t, ids in enumerate(ids_per_frame):
        for i in ids[ids > 0]:
            frames_by_id.setdefault(int(i), []).append(t)

    new_label, rows, new_id = {}, [], 1
    for old_id, frames in sorted(frames_by_id.items()):
        parent, start = 0, frames[0]
        for k in range(1, len(frames) + 1):
            if k == len(frames) or frames[k] != frames[k - 1] + 1:
                end = frames[k - 1]
                for t in range(start, end + 1):
                    new_label[(old_id, t)] = new_id
                rows.append([new_id, start, end, parent])
                parent, new_id = new_id, new_id + 1
                if k < len(frames):
                    start = frames[k]

    out = np.zeros(masks.shape, dtype=np.uint16)
    for t, ids in enumerate(ids_per_frame):
        lut = np.zeros(int(masks[t].max()) + 1, dtype=np.uint16)
        for i in ids[ids > 0]:
            lut[i] = new_label[(int(i), t)]
        out[t] = lut[masks[t]]
    return out, rows


def _write_ctc(masks, folder, prefix, track_file):
    """Save masks as one tif per frame plus a track file, in Cell Tracking Challenge format."""
    masks, rows = _split_gaps(masks)
    os.makedirs(folder, exist_ok=True)
    for t in range(masks.shape[0]):
        tifffile.imwrite(os.path.join(folder, f"{prefix}{t:03d}.tif"), masks[t])
    np.savetxt(os.path.join(folder, track_file),
               np.array(rows, dtype=int).reshape(-1, 4), fmt="%d")


def tracking_scores(gt_masks, tracked, region=Q2_TR):
    """Cell Tracking Challenge TRA (plus DET and LNK) using py-ctcmetrics.

    region: crop to score on, default the q2_tr test quadrant. Pass None for the whole frame.
    """
    from ctc_metrics import evaluate_sequence
    if region is not None:
        gt_masks = gt_masks[(slice(None),) + region]
        tracked = tracked[(slice(None),) + region]
    with tempfile.TemporaryDirectory() as tmp:
        res_dir = os.path.join(tmp, "01_RES")
        gt_dir = os.path.join(tmp, "01_GT")
        _write_ctc(tracked, res_dir, "mask", "res_track.txt")
        _write_ctc(gt_masks, os.path.join(gt_dir, "TRA"), "man_track", "man_track.txt")
        return evaluate_sequence(res_dir, gt_dir,
                                 metrics=["Valid", "TRA", "DET", "LNK"], threads=1)
