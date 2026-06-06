from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data     = "download_dataset/Annotations-1-enhanced/data.yaml",
    epochs   = 100,
    imgsz    = 512,
    batch    = 8,
    device   = "cpu",
    name     = "thesis-yolov11-zerodce",
    project  = "runs/train",
    patience = 100,
    plots    = True,
)