from ultralytics import YOLO

model = YOLO(r"C:\Users\Jastine\runs\detect\runs\train\thesis-yolov12-3\weights\best.pt")
metrics = model.val(data="download_dataset/Annotations-1/data.yaml")

print("mAP@50:", metrics.box.map50)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
