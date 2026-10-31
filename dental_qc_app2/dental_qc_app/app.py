import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from dental_qc import config
from dental_qc.pipeline import DentalQCPipeline
from dental_qc.report import prevalence_report, tooth_table

st.set_page_config(page_title="Dental QC Dashboard", layout="wide")
st.title("Automated Endodontic & Restorative Quality Assessment")
st.caption("Serial pipeline: Model 1 (FDI numbering) \u2192 Model 2 (RCT / restoration / periapical quality)")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Models")


def _weights_input(label, default_path, key):
    uploaded = st.sidebar.file_uploader(f"{label} weights (.pt)", type=["pt", "pth"], key=f"{key}_upload")
    if uploaded is not None:
        tmp_path = Path(tempfile.gettempdir()) / uploaded.name
        tmp_path.write_bytes(uploaded.getbuffer())
        return str(tmp_path)
    return st.sidebar.text_input(f"...or {label} weights path", value=default_path, key=f"{key}_path")


model1_path = _weights_input("Model 1 (FDI numbering)", config.MODEL1_WEIGHTS_PATH, "m1")
model2_path = _weights_input("Model 2 (RCT quality)", config.MODEL2_WEIGHTS_PATH, "m2")

model2_backend = st.sidebar.selectbox(
    "Model 2 backend",
    options=["yolo_obb", "yolo_classify", "yolo_detect", "pytorch_classifier"],
    index=["yolo_obb", "yolo_classify", "yolo_detect", "pytorch_classifier"].index(config.MODEL2_BACKEND),
    format_func=lambda x: {
        "pytorch_classifier": "PyTorch classifier (crops)",
        "yolo_classify": "Ultralytics YOLO classify (crops)",
        "yolo_detect": "Ultralytics YOLO detect (crops)",
        "yolo_obb": "Ultralytics YOLO OBB (crops)",
    }[x],
    help="Confirmed for this project's Model 2 checkpoint: task='obb', "
         "class 'Root Canal Treatment'.",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Running on: **{device}**")

conf_threshold = st.sidebar.slider("Model 1 detection confidence", 0.0, 1.0, config.MODEL1_CONF_THRESHOLD, 0.05)
config.MODEL1_CONF_THRESHOLD = conf_threshold


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline(m1_path, m2_path, dev, m2_backend):
    return DentalQCPipeline(m1_path, m2_path, dev, model2_backend=m2_backend).load()


# ---------------------------------------------------------------- main
uploaded_images = st.file_uploader(
    "Upload OPG(s) (panoramic X-rays)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_images and st.button(f"Run pipeline on {len(uploaded_images)} image(s)", type="primary"):
    try:
        pipeline = get_pipeline(model1_path, model2_path, device, model2_backend)
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()

    all_records = []
    tabs = st.tabs([f.name for f in uploaded_images])

    for file, tab in zip(uploaded_images, tabs):
        image = Image.open(file).convert("RGB")
        with st.spinner(f"Processing {file.name}..."):
            records = pipeline.run(image)
        all_records.append(records)

        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original OPG", use_container_width=True)
            with col2:
                heatmap = pipeline.build_heatmap(image, records)
                st.image(heatmap, caption="Periapical pathology heatmap", use_container_width=True)

            st.subheader("Per-tooth report")
            table = tooth_table(records)
            if table.empty:
                st.info("No teeth detected above the confidence threshold.")
            else:
                st.dataframe(table, use_container_width=True)

    st.divider()
    st.header("Dataset-level Dashboard")
    prevalence = prevalence_report(all_records)

    cols = st.columns(4)
    cols[0].metric("Images processed", prevalence["images_processed"])
    cols[1].metric("Teeth detected", prevalence["teeth_detected"])
    cols[2].metric("RCT prevalence", f'{prevalence["rct_prevalence_pct"] or 0}%')
    cols[3].metric("Periapical lesion prevalence", f'{prevalence["periapical_lesion_prevalence_pct"] or 0}%')

    with st.expander("Full prevalence breakdown"):
        st.json(prevalence)

    tables_by_image = [(f.name, tooth_table(recs)) for f, recs in zip(uploaded_images, all_records)]
    non_empty = [t.assign(Image=name) for name, t in tables_by_image if not t.empty]

    if non_empty:
        combined = pd.concat(non_empty, ignore_index=True)
        st.download_button(
            "Download combined report (CSV)",
            combined.to_csv(index=False).encode("utf-8"),
            file_name="dental_qc_report.csv",
            mime="text/csv",
        )
else:
    st.info("Upload one or more panoramic X-rays (OPGs) and click Run to generate the dashboard.")
