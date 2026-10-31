"""
Builds the two report outputs described in the project brief:
  - a per-tooth tabular report for a single OPG
  - prevalence statistics aggregated across a batch of OPGs
"""

from typing import List

import pandas as pd

from .pipeline import ToothRecord


def tooth_table(records: List[ToothRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {"Tooth (FDI)": rec.fdi_number, "Detection Conf.": round(rec.detection_confidence, 2)}
        for head, value in rec.findings.items():
            row[head.replace("_", " ").title()] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Tooth (FDI)").reset_index(drop=True)


def prevalence_report(all_records: List[List[ToothRecord]]) -> dict:
    """all_records: one list of ToothRecord per processed OPG."""
    flat = [rec for image_records in all_records for rec in image_records]
    n_images = len(all_records)
    n_teeth = len(flat)

    def count(pred_fn):
        return sum(1 for r in flat if pred_fn(r))

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    rct_present = count(lambda r: r.findings.get("rct_status", "").startswith("RCT_"))
    rct_adequate = count(lambda r: r.findings.get("rct_status") == "RCT_Adequate")
    rct_inadequate = count(lambda r: r.findings.get("rct_status") == "RCT_Inadequate")
    pal_present = count(lambda r: r.findings.get("pal_status") == "PAL_Present")
    rct_and_pal = count(
        lambda r: r.findings.get("rct_status", "").startswith("RCT_")
        and r.findings.get("pal_status") == "PAL_Present"
    )
    restoration_present = count(lambda r: r.findings.get("restoration_status", "").startswith("Restoration_"))
    restoration_adequate = count(lambda r: r.findings.get("restoration_status") == "Restoration_Adequate")
    restoration_inadequate = count(lambda r: r.findings.get("restoration_status") == "Restoration_Inadequate")

    return {
        "images_processed": n_images,
        "teeth_detected": n_teeth,
        "rct_treated_teeth": rct_present,
        "rct_prevalence_pct": pct(rct_present, n_teeth),
        "rct_adequate_pct_of_rct": pct(rct_adequate, rct_present),
        "rct_inadequate_pct_of_rct": pct(rct_inadequate, rct_present),
        "periapical_lesion_prevalence_pct": pct(pal_present, n_teeth),
        "rct_with_pal_pct_of_rct": pct(rct_and_pal, rct_present),
        "restoration_prevalence_pct": pct(restoration_present, n_teeth),
        "restoration_adequate_pct_of_restored": pct(restoration_adequate, restoration_present),
        "restoration_inadequate_pct_of_restored": pct(restoration_inadequate, restoration_present),
    }
