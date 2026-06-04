from ultralytics import YOLO

data_yaml = "download_dataset/Annotations-1/data.yaml"

model = YOLO("yolo12n.pt")

model.train(
    data=data_yaml,
    epochs=100,
    imgsz=512,
    batch=8,
    device="cpu",
    name="thesis-yolov12",
    project="runs/train"
)
