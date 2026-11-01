# Platelet Imaging Platform — Scaffold

Scaffold for the Direction A end-to-end platform (segmentation, tracking, evaluation,
visualisation) built on fine-tuned CellposeSAM.

## Status
Early scaffold. Model integration is stubbed out in `utils/model_utils.py` until
fine-tuning is complete — pages fall back to using the ground-truth mask as a
placeholder prediction so the UI can be built and tested now.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure
- `app.py` — landing page
- `pages/1_Upload_and_View.py` — load `.tif` image + `.npz`/`.tif` mask, frame navigation, overlay
- `pages/2_Segmentation.py` — run segmentation on a frame, compare to GT
- `pages/3_Metrics.py` — aggregate + per-cell Dice/IoU across frames (Hungarian-matched instances)
- `utils/io_utils.py` — file loading helpers
- `utils/model_utils.py` — model wrapper; set `MODEL_PATH` once training is done
- `utils/metrics_utils.py` — Dice/IoU (aggregate + per-cell)

## TODO once the model is trained
1. Set `MODEL_PATH` in `utils/model_utils.py` to the saved CellposeSAM checkpoint.
2. Re-test the Segmentation and Metrics pages with real predictions instead of the GT placeholder.
3. Add a Tracking page (cell-ID consistency across frames) — not yet built here.

## Prior work
Sellier (2025) began a Streamlit dashboard for model comparison but did not complete
it. Worth checking with Peter or Coco whether that code is still available before
extending this scaffold further — it may already cover ground this doesn't.
