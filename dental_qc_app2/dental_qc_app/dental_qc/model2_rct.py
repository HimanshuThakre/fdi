"""
Model 2 -- RCT / restoration / periapical quality classification.

Runs on the per-tooth crop produced from Model 1's detection box. Supports
single- or multi-head classifiers, as declared in config.MODEL2_HEADS.

Loading supports two checkpoint styles:
  1. A full pickled nn.Module (torch.save(model, path))          -> works as-is
  2. A state_dict, optionally wrapped in {"state_dict": ...}      -> fill in
     build_architecture() below with the exact class you trained.
"""

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import config


@dataclass
class ToothClassification:
    predictions: Dict[str, str] = field(default_factory=dict)    # head_name -> predicted class label
    confidences: Dict[str, float] = field(default_factory=dict)  # head_name -> confidence of that label


def build_architecture():
    """
    Only needed if MODEL2_WEIGHTS_PATH points to a state_dict rather than a
    full pickled model. Construct and return your model's architecture here, e.g.:

        import torchvision.models as tvm
        net = tvm.efficientnet_b0(weights=None)
        n_classes = sum(len(v) for v in config.MODEL2_HEADS.values())
        net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, n_classes)
        return net

    Leave this raising NotImplementedError if you always save full models
    (i.e. torch.save(model, path) rather than torch.save(model.state_dict(), path)).
    """
    raise NotImplementedError(
        "Model 2 checkpoint looks like a state_dict. Define build_architecture() "
        "in dental_qc/model2_rct.py to match the architecture you trained."
    )


class RCTQualityModel:
    def __init__(self, weights_path: str = None, device: str = None, backend: str = None):
        self.weights_path = weights_path or config.MODEL2_WEIGHTS_PATH
        self.device = device or config.DEVICE
        self.backend = backend or config.MODEL2_BACKEND
        self._model = None
        self._transform = transforms.Compose([
            transforms.Resize((config.MODEL2_IMG_SIZE, config.MODEL2_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.MODEL2_NORMALIZE_MEAN, config.MODEL2_NORMALIZE_STD),
        ])

    def load(self):
        if self._model is not None:
            return self

        if self.backend in ("yolo_classify", "yolo_detect", "yolo_obb"):
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError(
                    f"ultralytics is required for MODEL2_BACKEND='{self.backend}' "
                    "(`pip install ultralytics`)."
                ) from e
            self._model = YOLO(self.weights_path)
            return self

        # weights_only=False: we intend to support full pickled nn.Module checkpoints,
        # which newer PyTorch versions no longer unpickle by default. Only point this
        # at .pt files you trained/trust yourself.
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model = build_architecture()
            model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)
        self._model = model
        return self

    def predict(self, crop: Image.Image) -> ToothClassification:
        if self._model is None:
            self.load()

        if self.backend == "yolo_classify":
            return self._predict_yolo_classify(crop)
        if self.backend == "yolo_detect":
            return self._predict_yolo_detect(crop)
        if self.backend == "yolo_obb":
            return self._predict_yolo_obb(crop)
        return self._predict_pytorch(crop)

    def _predict_yolo_classify(self, crop: Image.Image) -> ToothClassification:
        results = self._model.predict(source=crop, device=self.device, verbose=False)[0]
        probs = results.probs  # Ultralytics classification result

        top_idx = int(probs.top1)
        label = results.names[top_idx]
        label = config.MODEL2_YOLO_CLASS_MAP.get(label, label)
        confidence = float(probs.top1conf)

        result = ToothClassification()
        result.predictions[config.MODEL2_YOLO_HEAD_NAME] = label
        result.confidences[config.MODEL2_YOLO_HEAD_NAME] = confidence
        return result

    def _predict_yolo_detect(self, crop: Image.Image) -> ToothClassification:
        results = self._model.predict(
            source=crop, conf=config.MODEL2_CONF_THRESHOLD, device=self.device, verbose=False,
        )[0]

        result = ToothClassification()

        if len(results.boxes) == 0:
            label = config.MODEL2_YOLO_NO_DETECTION_LABEL
            if label:
                result.predictions[config.MODEL2_YOLO_HEAD_NAME] = label
                result.confidences[config.MODEL2_YOLO_HEAD_NAME] = 1.0
            return result

        # multiple boxes can fire inside one tooth crop -- keep the most confident one
        best_box = max(results.boxes, key=lambda b: float(b.conf.item()))
        cls_idx = int(best_box.cls.item())
        names = results.names
        label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
        label = config.MODEL2_YOLO_CLASS_MAP.get(label, label)

        result.predictions[config.MODEL2_YOLO_HEAD_NAME] = label
        result.confidences[config.MODEL2_YOLO_HEAD_NAME] = float(best_box.conf.item())
        return result

    def _predict_yolo_obb(self, crop: Image.Image) -> ToothClassification:
        """OBB (oriented bounding box) mode: e.g. `yolo obb train`. Results carry
        their detections in `results.obb` instead of `results.boxes`/`.probs`."""
        results = self._model.predict(
            source=crop, conf=config.MODEL2_CONF_THRESHOLD, device=self.device, verbose=False,
        )[0]

        result = ToothClassification()
        obb = results.obb

        if obb is None or len(obb) == 0:
            label = config.MODEL2_YOLO_NO_DETECTION_LABEL
            if label:
                result.predictions[config.MODEL2_YOLO_HEAD_NAME] = label
                result.confidences[config.MODEL2_YOLO_HEAD_NAME] = 1.0
            return result

        # multiple oriented boxes can fire inside one crop -- keep the most confident
        confs = obb.conf
        best_idx = int(confs.argmax().item())
        cls_idx = int(obb.cls[best_idx].item())
        names = results.names
        label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
        label = config.MODEL2_YOLO_CLASS_MAP.get(label, label)

        result.predictions[config.MODEL2_YOLO_HEAD_NAME] = label
        result.confidences[config.MODEL2_YOLO_HEAD_NAME] = float(confs[best_idx].item())
        return result

    def _predict_pytorch(self, crop: Image.Image) -> ToothClassification:
        x = self._transform(crop.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_output = self._model(x)

        # Normalize the model's output into {head_name: logits_tensor}. Supports
        # models that return a single tensor, a tuple of tensors (one per head, in
        # config.MODEL2_HEADS order), or a dict already keyed by head name.
        head_names = list(config.MODEL2_HEADS.keys())

        if isinstance(raw_output, dict):
            head_logits = raw_output
        elif isinstance(raw_output, (tuple, list)):
            head_logits = dict(zip(head_names, raw_output))
        else:
            head_logits = {head_names[0]: raw_output}

        result = ToothClassification()
        for head_name, class_names in config.MODEL2_HEADS.items():
            if not class_names or head_name not in head_logits:
                continue
            probs = F.softmax(head_logits[head_name], dim=1)[0]
            top_idx = int(torch.argmax(probs).item())
            result.predictions[head_name] = class_names[top_idx]
            result.confidences[head_name] = float(probs[top_idx].item())

        return result
