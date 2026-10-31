# Dental QC Dashboard

A Streamlit app implementing the serial pipeline from the project brief
("Automated Endodontic & Restorative Quality Assessment"):

```
OPG image
   |
   v
Model 1: FDI numbering (detects + numbers every tooth)
   |
   v
crop each detected tooth, with padding
   |
   v
Model 2: RCT / restoration / periapical quality classification (per tooth)
   |
   v
combine into one record per tooth
   |
   v
Dashboard: heatmap + per-tooth table + prevalence stats
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put your two trained checkpoints in `models/`:

```
models/fdi_numbering.pt   <- Model 1
models/rct_quality.pt     <- Model 2
```

(Or leave them wherever they are — the app's sidebar lets you either upload a
`.pt`/`.pth` file directly or type a path, per run. **For large checkpoints,
prefer typing the path** — the upload box goes through the browser and is
capped at 2GB via `.streamlit/config.toml` [raise it further if you need to],
while a typed path is read straight off disk with no size limit.)

```bash
streamlit run app.py
```

## Project layout

```
app.py                    Streamlit UI: upload, run, display dashboard
dental_qc/
  config.py               All tunable settings — start here
  model1_fdi.py            Model 1 wrapper (detection + FDI numbering)
  model2_rct.py             Model 2 wrapper (per-tooth quality classification)
  pipeline.py              Orchestrates Model 1 -> crop -> Model 2 -> combine
  report.py                Per-tooth table + dataset-level prevalence stats
models/                   Put your .pt weight files here
```

## Wiring in your two real models

The app runs and produces the full dashboard layout without any weights, but
you'll need to check/adjust three things in `dental_qc/config.py` and the two
model wrapper files so the numbers are actually correct for your checkpoints.

### 1. Model 1 — FDI numbering (`model1_fdi.py`)

Assumed to be an **Ultralytics YOLO** `.pt` checkpoint (this is the standard
`.pt` extension for YOLOv5/v8/v9/v11 and is by far the most common setup for
this task in dental AI work — e.g. the DENTEX-style tooth detection +
numbering pipelines). If that's what you have, nothing needs to change beyond
pointing `MODEL1_WEIGHTS_PATH` at your file.

- If your checkpoint's own class names are already `"11"`..`"48"`, detection
  labels are used directly.
- If your checkpoint uses generic class indices (`0`..`31`), edit
  `config.FDI_CLASSES` so its **order matches your training class order** —
  that list is what maps index → FDI number.
- If Model 1 is *not* YOLO (e.g. a custom detector, Faster R-CNN, DETR),
  replace the body of `FDINumberingModel.load()` / `.predict()` — everything
  downstream only depends on `predict()` returning a list of `ToothDetection`
  objects (`fdi_number`, `confidence`, `box`).

### 2. Model 2 — RCT / restoration / periapical quality (`model2_rct.py`)

**For this project specifically: Model 2 is confirmed to be an Ultralytics
OBB (oriented bounding box) checkpoint — `task="obb"`, single class
`"Root Canal Treatment"`** — trained with `yolo obb train`. This is now the
default (`MODEL2_BACKEND = "yolo_obb"`), pre-configured with the exact class
name mapped to `"RCT_Present"`, so it should work without changing anything
in the sidebar. Since this model only detects RCT *presence* and doesn't
distinguish adequate vs. inadequate, the adequate/inadequate prevalence
stats will show 0 / not applicable — that's expected, not a bug, unless a
future model version adds that distinction.

Four backends are available in total, toggled from the app's sidebar (or
`config.MODEL2_BACKEND`), in case Model 2 gets swapped for a different
checkpoint later:

- **`yolo_obb`** (current default) — oriented-box detector, as above.
- **`yolo_classify`** — an Ultralytics checkpoint trained with
  `yolo classify train` (whole-crop classification, no boxes at all).
- **`yolo_detect`** — an Ultralytics checkpoint trained with
  `yolo detect train` (ordinary axis-aligned boxes).
- **`pytorch_classifier`** — a plain PyTorch image classifier for anything
  that isn't a YOLO checkpoint at all.
  - **Checkpoint format**: if you saved with `torch.save(model, path)` (a
    full pickled model), loading works as-is. If you saved a `state_dict`
    (`torch.save(model.state_dict(), path)`), fill in `build_architecture()`
    at the top of `model2_rct.py` with the exact architecture class you
    trained, so the state dict has something to load into.

If you're ever unsure which of the three Ultralytics task types a `.pt` file
actually is, this settles it directly (no trial-and-error needed):

```python
from ultralytics import YOLO
m = YOLO("path/to/model.pt")
print(m.task)   # "detect" | "classify" | "segment" | "obb" | "pose"
print(m.names)  # its class names
```

`inspect_checkpoint.py` (see below) also flags a checkpoint as Ultralytics
automatically, but doesn't run this task check — the snippet above is the
fastest way to pin down which of the three YOLO backends applies.

All three YOLO backends share `config.MODEL2_YOLO_CLASS_MAP` to rename
YOLO's own class names to this app's canonical labels, and store their
result under the single head named `config.MODEL2_YOLO_HEAD_NAME` (default
`"rct_status"`) — each of the three modes only predicts one thing per crop.
- **Output heads**: `config.MODEL2_HEADS` declares what the model predicts.
  As shipped it assumes three heads (`rct_status`, `restoration_status`,
  `pal_status`) matching the brief's three quality dimensions. If your model
  only predicts RCT status, delete the other two entries — the report and
  dashboard adapt automatically and just omit those columns/stats. The
  wrapper also accepts a model that returns a single tensor (one head), a
  tuple of tensors (multiple heads, in `MODEL2_HEADS` order), or a dict
  keyed by head name — whichever your model actually does.
- **Preprocessing**: `MODEL2_IMG_SIZE` and the normalization mean/std need to
  match what you trained with (defaults are ImageNet stats, a common
  transfer-learning starting point).

### 3. Class label conventions

The report/prevalence logic pattern-matches on specific string values —
`"RCT_Adequate"`, `"RCT_Inadequate"`, `"PAL_Present"`, etc. If you rename the
classes in `config.MODEL2_HEADS`, update the matching strings in
`dental_qc/report.py` and `dental_qc/pipeline.py::build_heatmap` to match.

## Notes / known simplifications

- **Heatmap**: the "periapical pathology heatmap" is a box-level risk overlay
  (inadequate RCT / present PAL / inadequate restoration → higher weight,
  blurred with a Gaussian filter), not true pixel-level model attention. If
  you want a real Grad-CAM-style map from Model 2's own activations, that's a
  natural extension point in `DentalQCPipeline.build_heatmap`.
- **Primary (deciduous) teeth**: `FDI_CLASSES` only lists the 32 permanent
  teeth. Extend it with the 51–85 range if your data includes children's OPGs.
- **Batching**: the app processes uploaded images one at a time in a loop.
  For genuinely large datasets (the brief mentions sets of ~1000 OPGs), you'll
  likely want a separate batch/CLI script reusing `DentalQCPipeline` rather
  than uploading 1000 files through the browser — happy to help build that
  next if useful.
