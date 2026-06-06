"""
Trains all 4 models in sequence:
1. YOLOv11 Baseline
2. YOLOv11 + Zero-DCE
3. YOLOv12 Baseline
4. YOLOv12 + Zero-DCE
"""
from ultralytics import YOLO

BASELINE_DATA = "download_dataset/Annotations-1/data.yaml"
ENHANCED_DATA = "download_dataset/Annotations-1-enhanced/data.yaml"

COMMON = dict(
    epochs   = 100,
    imgsz    = 512,
    batch    = 8,
    device   = "cpu",
    patience = 100,
    plots    = True,
)

# ── 1. YOLOv11 Baseline ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training 1/4: YOLOv11 Baseline")
print("="*60)
YOLO("yolo11n.pt").train(
    data    = BASELINE_DATA,
    name    = "yolov11_baseline",
    project = "runs/train",
    **COMMON
)

# ── 2. YOLOv11 + Zero-DCE ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training 2/4: YOLOv11 + Zero-DCE")
print("="*60)
YOLO("yolo11n.pt").train(
    data    = ENHANCED_DATA,
    name    = "yolov11_zerodce",
    project = "runs/train",
    **COMMON
)

# ── 3. YOLOv12 Baseline ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training 3/4: YOLOv12 Baseline")
print("="*60)
YOLO("yolo12n.pt").train(
    data    = BASELINE_DATA,
    name    = "yolov12_baseline",
    project = "runs/train",
    **COMMON
)

# ── 4. YOLOv12 + Zero-DCE ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training 4/4: YOLOv12 + Zero-DCE")
print("="*60)
YOLO("yolo12n.pt").train(
    data    = ENHANCED_DATA,
    name    = "yolov12_zerodce",
    project = "runs/train",
    **COMMON
)

print("\n" + "="*60)
print("All 4 models done! Now run: python compare_results.py")
print("="*60)