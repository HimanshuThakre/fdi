from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # corrected: medium, detection

model.train(
    data="./data/data.yaml",
    epochs=100,
    imgsz=1024,
    batch=8,
    device=0,
    cache=True,          # ← added

    save=True,
    save_period=10,
    patience=20,

    fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0,
    degrees=5.0, translate=0.05, scale=0.2,
    hsv_v=0.3, hsv_s=0.0, hsv_h=0.0,

    project="runs",
    name="fdi_v1",
    plots=True,
    val=True
)