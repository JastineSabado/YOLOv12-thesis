import os
import shutil
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from zerodce import ZeroDCE

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DCE_WEIGHTS = "Zero-DCE_weights.pth"
SRC_BASE    = "download_dataset/Annotations-1"
DST_BASE    = "download_dataset/Enhanced"

print(f"Using device: {DEVICE}")

# Load Zero-3DCE
dce_model = ZeroDCE().to(DEVICE)
dce_model.load_state_dict(torch.load(DCE_WEIGHTS, map_location=DEVICE))
dce_model.eval()
to_tensor = transforms.ToTensor()

def enhance_split(split):
    src_img = os.path.join(SRC_BASE, split, "images")
    src_lbl = os.path.join(SRC_BASE, split, "labels")
    dst_img = os.path.join(DST_BASE, split, "images")
    dst_lbl = os.path.join(DST_BASE, split, "labels")

    if not os.path.exists(src_img):
        print(f"[{split}] not found, skipping.")
        return

    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    files = [f for f in os.listdir(src_img)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\n[{split}] Enhancing {len(files)} images...")

    for i, fname in enumerate(files, 1):
        # Enhance image with Zero-3DCE
        img    = Image.open(os.path.join(src_img, fname)).convert("RGB")
        tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            enhanced = dce_model(tensor)
        enhanced = enhanced.squeeze(0).cpu().numpy()
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        Image.fromarray(enhanced.transpose(1, 2, 0)).save(
            os.path.join(dst_img, fname))

        # Copy label as-is (bounding boxes don't change)
        label_file    = os.path.splitext(fname)[0] + ".txt"
        src_lbl_path  = os.path.join(src_lbl, label_file)
        if os.path.exists(src_lbl_path):
            shutil.copy(src_lbl_path, os.path.join(dst_lbl, label_file))

        print(f"  [{i}/{len(files)}] {fname}")

    print(f"[{split}] Done.")

# Enhance all splits
for split in ["train", "valid", "test"]:
    enhance_split(split)

# Copy and update data.yaml
src_yaml = os.path.join(SRC_BASE, "data.yaml")
dst_yaml = os.path.join(DST_BASE, "data.yaml")
os.makedirs(DST_BASE, exist_ok=True)

with open(src_yaml, "r") as f:
    content = f.read()
content = content.replace(SRC_BASE, DST_BASE)
with open(dst_yaml, "w") as f:
    f.write(content)

print(f"\nDone! Enhanced dataset saved to: {DST_BASE}")