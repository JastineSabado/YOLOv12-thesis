from ultralytics import YOLO

data_yaml = "download_dataset/Enhanced/data.yaml"

model = YOLO("yolo12n.pt")

model.train(
    data=data_yaml,
    epochs=100,
    imgsz=512,
    batch=8,
    device="cpu",    # change to "cuda" if you have a GPU
    name="thesis-yolov12-zerodce",
    project="runs/train"
)

