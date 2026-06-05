import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from zerodce import ZeroDCE

VIDEO_PATH   = "your_video.mp4"  # change to your video path or 0 for webcam
YOLO_WEIGHTS = "runs/train/thesis-yolov12-zerodce/weights/best.pt"
DCE_WEIGHTS  = "Zero-DCE_weights.pth"
CONF         = 0.4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
dce = ZeroDCE().to(DEVICE)
dce.load_state_dict(torch.load(DCE_WEIGHTS, map_location=DEVICE))
dce.eval()

yolo      = YOLO(YOLO_WEIGHTS)
to_tensor = transforms.ToTensor()

def enhance(frame_bgr):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = dce(tensor)
    out = out.squeeze(0).cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    enhanced  = enhance(frame)             # Step 1: Zero-3DCE
    results   = yolo(enhanced, conf=CONF)  # Step 2: YOLOv12
    annotated = results[0].plot()          # Step 3: Draw boxes

    # Show original and enhanced+detected side by side
    side = cv2.hconcat([frame, annotated])
    cv2.imshow("Original  |  Zero-3DCE + YOLOv12", side)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()