import cv2
import csv
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

FDI_ORDER = ['11','12','13','14','15','16','17','18',
             '21','22','23','24','25','26','27','28',
             '31','32','33','34','35','36','37','38',
             '41','42','43','44','45','46','47','48']

def preprocess(image_path, target_size=1024):
    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas

def remove_duplicates(boxes, class_names):
    """Keep only the highest-confidence box per FDI class."""
    best = {}
    duplicates = []
    for box in boxes:
        fdi = class_names[int(box.cls)]
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        if fdi not in best or conf > best[fdi]["confidence"]:
            if fdi in best:
                duplicates.append(fdi)
            best[fdi] = {"confidence": conf, "bbox": xyxy}
    return best, duplicates

def sort_by_fdi_order(detections):
    return [(fdi, detections[fdi]) for fdi in FDI_ORDER if fdi in detections]

def detect_missing_teeth(detections):
    return [fdi for fdi in FDI_ORDER if fdi not in detections]

def generate_report(sorted_detections, missing_teeth, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FDI_Number", "Status", "Confidence", "X1", "Y1", "X2", "Y2"])
        for fdi, d in sorted_detections:
            writer.writerow([fdi, "Present", round(d["confidence"], 3), *[round(c, 1) for c in d["bbox"]]])
        for fdi in missing_teeth:
            writer.writerow([fdi, "Missing", "", "", "", "", ""])

def draw_annotated_image(image, sorted_detections, out_path):
    img = image.copy()
    for fdi, d in sorted_detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, fdi, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)

os.makedirs("outputs", exist_ok=True)

model = YOLO("runs/fdi_v1/weights/best.pt")
raw_image_path = "inference/customer_xray.jpg"
img = preprocess(raw_image_path)

results = model.predict(source=img, conf=0.4, iou=0.4, imgsz=1024)
result = results[0]

detections, duplicates = remove_duplicates(result.boxes, result.names)
if duplicates:
    print(f"⚠️ Duplicate FDI class(es) resolved by keeping highest confidence: {duplicates}")

sorted_detections = sort_by_fdi_order(detections)
missing = detect_missing_teeth(detections)

print("\n===== FDI Tooth Report =====")
for fdi, d in sorted_detections:
    print(f"Tooth {fdi}: present (confidence {d['confidence']:.2f})")
for fdi in missing:
    print(f"Tooth {fdi}: MISSING")

generate_report(sorted_detections, missing, "outputs/report.csv")
draw_annotated_image(img, sorted_detections, "outputs/predicted_output.jpg")
print("\nSaved → outputs/report.csv, outputs/predicted_output.jpg")