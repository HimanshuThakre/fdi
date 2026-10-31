# FDI — Dental OPG tooth numbering and RCT dashboard

Takes a panoramic dental X-ray (OPG) and produces a "Dental Health Dashboard":
FDI tooth numbers, root-canal-treatment (RCT) status per tooth, a box-level risk
heatmap, a per-tooth table, a missing-teeth list, CSV export and batch prevalence
statistics.

> This README describes only what is in this repository. Places where the code
> and the intended design disagree are listed under [Known gaps](#known-gaps).

## What is used

### Models

| | Model 1 – FDI numbering | Model 2 – RCT |
|---|---|---|
| Framework | Ultralytics YOLO (`detect`) | Ultralytics YOLO (`obb`, default backend) |
| Classes | 32 permanent teeth, `"11"`…`"48"` (`fdi-tooth-detection/data/data.yaml`) | 1 class: `Root Canal Treatment` |
| Input | Whole OPG | Crop of each tooth found by Model 1 (15% padding) |
| Output | One box + FDI number + confidence per tooth | No box → `No_RCT`; box → `RCT_Present` |
| Weights | `models/fdi_numbering.pt` (not in repo) | `models/rct_quality.pt` (not in repo) |

Model 2 detects RCT **presence only**. It cannot say adequate vs inadequate, and
there is **no model** for periapical lesions (PAL) or restorations — those
columns are declared in `config.py` but not predicted.

Other Model 2 backends exist in `model2_rct.py` for swapping checkpoints later:
`yolo_detect`, `yolo_classify`, `pytorch_classifier`.

### Libraries

Model 1 training/inference (`fdi-tooth-detection/requirements.txt`, unpinned):
`ultralytics`, `opencv-python`, `Pillow`, `numpy`, `torch`, `torchvision`.

Dashboard app (`dental_qc_app2/dental_qc_app/requirements.txt`):
`streamlit>=1.35`, `torch>=2.1`, `torchvision>=0.16`, `ultralytics>=8.2`,
`pandas>=2.0`, `numpy>=1.24`, `scipy>=1.11`, `matplotlib>=3.8`, `Pillow>=10.0`.

### Training

`fdi-tooth-detection/train.py` (as committed): YOLOv8m (`yolov8m.pt`), 100 epochs,
`imgsz=1024`, `batch=8`, `patience=20`, `save_period=10`. Augmentation:
`fliplr=0`, `flipud=0`, `mosaic=0`, `mixup=0`, `degrees=5`, `translate=0.05`,
`scale=0.2`, `hsv_v=0.3`, `hsv_s=0`, `hsv_h=0`. Output: `runs/fdi_v1`.

Dataset layout is described by `fdi-tooth-detection/data/data.yaml`
(`train/images`, `valid/images`, `test/images`). The images themselves are not
in the repo.

### Inference for Model 1 (`fdi-tooth-detection/predict.py`)

1. Load image, convert to RGB.
2. Resize keeping aspect ratio onto a black 1024×1024 canvas.
3. `model.predict(imgsz=1024, conf=0.4, iou=0.4)`.
4. Keep only the highest-confidence box per FDI number.
5. Sort in FDI order, list missing teeth, write `outputs/<name>_report.csv`
   and an annotated image.

### Hardware / environment

- Training and prediction were run on a Tesla V100 32 GB with VS Code
  Remote-SSH (from the project history; not recorded in the repo files).
- The Streamlit app is configured with `DEVICE = "cuda"` and falls back to CPU.
  The brief describes the app host as Ubuntu, CPU only, venv `~/dental_venv`.
- Tools: Git, VS Code (`.vscode/tasks.json` is only a C/C++ build task and is
  unrelated to this project), Google Colab for the notebooks.

### Notebooks (read from imports and cell headings only)

| Notebook | What it contains |
|---|---|
| `first.ipynb`, `Untitled0 (4).ipynb` | Colab; Ultralytics YOLO training (`yolov8n.pt`, 50 epochs, `imgsz=640`, `batch=16`); the latter also a `imgsz=1280`, `batch=4` run |
| `ZERO.ipynb` | Colab; several YOLOv8 n/s/m runs at 640 and 1280, plus `ensemble-boxes` weighted box fusion |
| `lastodcolab.ipynb` | Colab; YOLOv8n training (80 epochs, 640) and many evaluation/prediction cells at `imgsz=1280`; the main training pipeline |
| `dental_opg_resnet50 (1).ipynb` | Experiment: custom Faster R-CNN with ResNet50 backbone (PyTorch/torchvision, YOLO-format labels) |

Results and metrics inside the notebooks have not been summarised here.

## Install and run

Model 1 only:

```bash
cd fdi-tooth-detection
pip install -r requirements.txt
# put images in inference/, weights at the path set in predict.py
python predict.py
```

Dashboard:

```bash
cd dental_qc_app2/dental_qc_app
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# put fdi_numbering.pt and rct_quality.pt in models/ (or type paths in the sidebar)
streamlit run app.py
```

`inspect_checkpoint.py` reports which Ultralytics task a `.pt` file is.
Typed sidebar paths avoid the browser upload limit for large checkpoints.

## Weights

`.pt` files are not committed (GitHub rejects files over 100 MB). Add `*.pt` to
`.gitignore`, or use Git LFS if the weights must travel with the repo.

## Known gaps

These are differences between files in this repo. Nothing was changed; model
inference settings are only changed after you confirm.

1. **App Model 1 inference differs from `predict.py`.** `dental_qc/config.py`
   has `MODEL1_IMG_SIZE = 640`, `MODEL1_IOU_THRESHOLD = 0.45`,
   `MODEL1_CONF_THRESHOLD = 0.25`, and `model1_fdi.py` passes the image straight
   to YOLO with no black letterbox and no one-box-per-FDI-number step.
   `predict.py` uses 1024 / 0.4 and both of those. Per the brief, repeated FDI
   numbers came from running at 640.
2. **Committed `predict.py` and `train.py` are older than the versions used in
   the latest runs**: no EXIF auto-orient, letterbox pinned to the top-left,
   `conf=0.4`, no confidence bands, no overlap trimming.
3. **`.streamlit/config.toml` is missing**, although the app README refers to it.
4. **`__pycache__/*.pyc` files are committed**; they should be in `.gitignore`.
5. The app README text mentions a PAL/restoration classifier; no such model exists.
6. `fdi-tooth-detection/inference/` contains patient-style OPG images. Check
   that they may be public.
