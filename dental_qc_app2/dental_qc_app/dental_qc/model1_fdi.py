"""
Model 1 -- FDI tooth numbering.

Wraps an Ultralytics YOLO (.pt) detector. If your model isn't a YOLO
checkpoint, swap the body of `FDINumberingModel.load()` and `.predict()` --
the rest of the pipeline only depends on the `ToothDetection` list returned
by `predict()`, documented below.
"""

from dataclasses import dataclass
from typing import List

from PIL import Image

from . import config


@dataclass
class ToothDetection:
    fdi_number: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixel coordinates on the original image


class FDINumberingModel:
    def __init__(self, weights_path: str = None, device: str = None):
        self.weights_path = weights_path or config.MODEL1_WEIGHTS_PATH
        self.device = device or config.DEVICE
        self._model = None

    def load(self):
        if self._model is not None:
            return self

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for Model 1 (`pip install ultralytics`). "
                "If Model 1 isn't a YOLO checkpoint, replace the body of "
                "FDINumberingModel.load() / .predict() with your own "
                "architecture's loading and inference code instead."
            ) from e

        self._model = YOLO(self.weights_path)
        return self

    def predict(self, image: Image.Image) -> List[ToothDetection]:
        """Run detection on one PIL image; return one ToothDetection per tooth found."""
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=image,
            conf=config.MODEL1_CONF_THRESHOLD,
            iou=config.MODEL1_IOU_THRESHOLD,
            imgsz=config.MODEL1_IMG_SIZE,
            device=self.device,
            verbose=False,
        )[0]

        names = results.names  # class index -> label, as stored in the checkpoint
        detections = []

        for box in results.boxes:
            cls_idx = int(box.cls.item())
            label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)

            # If your checkpoint's own class names are already "11".."48", this is a
            # no-op. If they're generic indices, map through FDI_CLASSES instead.
            if label not in config.FDI_CLASSES and 0 <= cls_idx < len(config.FDI_CLASSES):
                label = config.FDI_CLASSES[cls_idx]

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                ToothDetection(
                    fdi_number=label,
                    confidence=float(box.conf.item()),
                    box=(x1, y1, x2, y2),
                )
            )

        return detections
