"""
Central configuration for the Dental QC pipeline.

Edit the values below to match your two trained models. Nothing else in the
codebase should need to change for a typical swap-in of new weights.
"""

# ----------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------
DEVICE = "cuda"  # app.py auto-falls back to "cpu" if no GPU is available

# ----------------------------------------------------------------------
# Model 1 -- FDI Numbering (tooth detection + FDI classification)
# Default assumption: an Ultralytics YOLO (.pt) checkpoint trained on OPGs,
# with one detection class per FDI tooth number.
# ----------------------------------------------------------------------
MODEL1_WEIGHTS_PATH = "models/fdi_numbering.pt"  # default path; overridable in the sidebar
MODEL1_CONF_THRESHOLD = 0.25
MODEL1_IOU_THRESHOLD = 0.45
MODEL1_IMG_SIZE = 640

# Standard FDI two-digit notation for permanent (adult) teeth, by quadrant.
# This is ONLY used as a fallback to map generic class indices (0..31) to FDI
# numbers if your checkpoint's own class names aren't already "11".."48".
# The order here must match your model's class-index order if it's used.
# Extend with primary teeth (51-85) if your model was trained on those too.
FDI_CLASSES = [
    "18", "17", "16", "15", "14", "13", "12", "11",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "48", "47", "46", "45", "44", "43", "42", "41",
    "31", "32", "33", "34", "35", "36", "37", "38",
]

# ----------------------------------------------------------------------
# Model 2 -- RCT / Restoration / Periapical quality classification
# Runs on the per-tooth crop produced from Model 1's detection box.
# Supports single-head or multi-head classifiers -- declare each head below.
# Comment out (or remove) a head your model doesn't predict; the report
# simply omits that column instead of erroring.
# ----------------------------------------------------------------------
MODEL2_WEIGHTS_PATH = "models/rct_quality.pt"
MODEL2_IMG_SIZE = 224
MODEL2_CROP_PADDING = 0.15  # extra margin around each Model-1 box, as a fraction of box size

# "pytorch_classifier": a plain PyTorch model run on each crop.
# "yolo_classify": Ultralytics YOLO checkpoint trained with `yolo classify train`.
# "yolo_detect": Ultralytics YOLO checkpoint trained with `yolo detect train`.
# "yolo_obb": Ultralytics YOLO checkpoint trained with `yolo obb train`
#   (oriented bounding boxes) -- confirmed via `model.task` to be Model 2's
#   actual format (task="obb", single class "Root Canal Treatment"), so this
#   is now the default.
MODEL2_BACKEND = "yolo_obb"

# Only used when MODEL2_BACKEND in ("yolo_detect", "yolo_obb").
MODEL2_CONF_THRESHOLD = 0.25
# Label to record when no box at all is detected in the crop (e.g. no RCT
# filling visible). Set to None to leave that head blank instead.
MODEL2_YOLO_NO_DETECTION_LABEL = "No_RCT"

# Only used when MODEL2_BACKEND is one of the three yolo_* backends. YOLO
# classify/detect/obb modes each predict one thing at a time; this is the
# head that result gets stored under.
MODEL2_YOLO_HEAD_NAME = "rct_status"

# Maps YOLO's own class names to this app's canonical labels. Pre-filled to
# match the confirmed class name of this project's actual Model 2 checkpoint.
# Since this model only detects presence (not adequacy), it maps to a
# presence-only label -- prevalence stats and the heatmap both handle
# "RCT_Present" already (they just can't break it down into adequate vs.
# inadequate the way a model with that distinction would let them).
MODEL2_YOLO_CLASS_MAP = {
    "Root Canal Treatment": "RCT_Present",
}

MODEL2_HEADS = {
    "rct_status": ["No_RCT", "RCT_Adequate", "RCT_Inadequate"],
    "restoration_status": ["No_Restoration", "Restoration_Adequate", "Restoration_Inadequate"],
    "pal_status": ["PAL_Absent", "PAL_Present"],
}

# ImageNet normalization -- change if your classifier was trained differently
MODEL2_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
MODEL2_NORMALIZE_STD = [0.229, 0.224, 0.225]
