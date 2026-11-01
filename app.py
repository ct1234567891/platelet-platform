"""AI-Empowered Platelet Imaging Platform -- Home

Run with:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Platelet Imaging Platform", layout="wide")

st.title("AI-Empowered Platelet Imaging Platform")


st.markdown("""
This is a scaffold for the end-to-end platform. Use the pages in the sidebar:

1. **Upload & View** -- load a `.tif` image stack and its matching mask (`.npz` or `.tif`),
   step through frames, and toggle the mask overlay.
2. **Segmentation** -- run the fine-tuned CellposeSAM model on a frame (falls back to a
   placeholder using the loaded GT mask until `utils/model_utils.py -> MODEL_PATH` is set).
3. **Metrics** -- aggregate and per-cell Dice / IoU across frames, benchmarked against
   the Sellier (2025) targets (Dice > 0.90, AP > 0.80 @ IoU 0.50, Tracking Accuracy > 0.90).

**Status:** scaffold stage -- wire in the trained model path once fine-tuning is complete.
Tracking (cell-ID consistency across frames) is not yet built as a page here; it's the
natural next addition once the model is trained.
""")


