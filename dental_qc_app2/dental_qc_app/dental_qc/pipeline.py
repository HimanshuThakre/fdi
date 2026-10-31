"""
Serial pipeline: Model 1 (FDI numbering) -> crop each detected tooth ->
Model 2 (RCT / restoration / periapical quality) -> combine into one
record per tooth.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from PIL import Image

from . import config
from .model1_fdi import FDINumberingModel, ToothDetection
from .model2_rct import RCTQualityModel


@dataclass
class ToothRecord:
    fdi_number: str
    box: tuple
    detection_confidence: float
    findings: Dict[str, str] = field(default_factory=dict)         # e.g. {"rct_status": "RCT_Inadequate"}
    finding_confidences: Dict[str, float] = field(default_factory=dict)


def _pad_box(box, padding_frac, image_size):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * padding_frac, h * padding_frac
    img_w, img_h = image_size
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


class DentalQCPipeline:
    """Loads both models once, then processes one OPG at a time."""

    def __init__(self, model1_weights=None, model2_weights=None, device=None, model2_backend=None):
        self.model1 = FDINumberingModel(model1_weights, device)
        self.model2 = RCTQualityModel(model2_weights, device, backend=model2_backend)

    def load(self):
        self.model1.load()
        self.model2.load()
        return self

    def run(self, image: Image.Image) -> List[ToothRecord]:
        """Run the full serial pipeline on one OPG; return one record per detected tooth."""
        detections: List[ToothDetection] = self.model1.predict(image)

        records = []
        for det in detections:
            padded_box = _pad_box(det.box, config.MODEL2_CROP_PADDING, image.size)
            crop = image.crop(tuple(int(v) for v in padded_box))

            classification = self.model2.predict(crop)

            records.append(
                ToothRecord(
                    fdi_number=det.fdi_number,
                    box=det.box,
                    detection_confidence=det.confidence,
                    findings=classification.predictions,
                    finding_confidences=classification.confidences,
                )
            )

        return records

    def build_heatmap(self, image: Image.Image, records: List[ToothRecord]) -> Image.Image:
        """
        Overlay a risk heatmap on the OPG: each tooth's box region is weighted by a
        simple risk score (inadequate RCT, a present periapical lesion, or an
        inadequate restoration score higher), then blurred and colorized.

        This is a box-level stand-in, not true pixel-level model attention. If you
        want a real Grad-CAM-style map, hook it in here using Model 2's activations
        for the "pal_status" head instead of the fixed per-status weights below.
        """
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter

        w, h = image.size
        heat = np.zeros((h, w), dtype=np.float32)

        for rec in records:
            score = 0.0
            status = rec.findings.get("rct_status", "")
            if status == "RCT_Inadequate":
                score += 0.7
            elif status == "RCT_Adequate":
                score += 0.2
            elif status.startswith("RCT_"):
                # a presence-only detector (e.g. yolo_obb) that doesn't distinguish
                # adequate vs. inadequate -- flag it, just less urgently than a
                # confirmed-inadequate finding
                score += 0.4
            if rec.findings.get("pal_status") == "PAL_Present":
                score += 0.8
            if rec.findings.get("restoration_status") == "Restoration_Inadequate":
                score += 0.5

            if score <= 0:
                continue

            x1, y1, x2, y2 = [int(v) for v in rec.box]
            heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], score)

        if heat.max() > 0:
            heat = gaussian_filter(heat, sigma=max(w, h) * 0.01)
            heat = heat / heat.max()

        cmap = plt.get_cmap("jet")
        colored = (cmap(heat)[:, :, :3] * 255).astype(np.uint8)
        heat_img = Image.fromarray(colored).convert("RGBA")
        heat_img.putalpha(Image.fromarray((heat * 160).astype(np.uint8)))

        base = image.convert("RGBA")
        return Image.alpha_composite(base, heat_img)
