"""
Trains Zero-DCE on your dataset images, then saves enhanced copies.
Run this ONCE before training YOLOv11.
"""
import os, glob, shutil, torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from zerodce import ZeroDCE, ZeroDCELoss

DATASET_ROOT  = "download_dataset/Annotations-1"
ENHANCED_ROOT = "download_dataset/Annotations-1-enhanced"
EPOCHS        = 10
BATCH_SIZE    = 8
LR            = 1e-4
IMG_SIZE      = 256
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

class ImageDataset(Dataset):
    def __init__(self, dirs):
        self.paths = []
        for d in dirs:
            self.paths += glob.glob(os.path.join(d, "*.jpg"))
            self.paths += glob.glob(os.path.join(d, "*.png"))
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.paths[i]

def train_zerodce(model, loader, crit, opt):
    model.train()
    for ep in range(EPOCHS):
        total = 0
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            enh, A = model(imgs)
            loss, *_ = crit(enh, imgs, A)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  Epoch [{ep+1}/{EPOCHS}]  loss: {total/len(loader):.4f}")

def enhance_and_save(model, split):
    img_dir = os.path.join(DATASET_ROOT, split, "images")
    if not os.path.exists(img_dir):
        print(f"  Skipping {split} (not found)")
        return

    out_dir = os.path.join(ENHANCED_ROOT, split, "images")
    os.makedirs(out_dir, exist_ok=True)

    paths = (glob.glob(os.path.join(img_dir, "*.jpg")) +
             glob.glob(os.path.join(img_dir, "*.png")))

    to_t = transforms.ToTensor()
    to_p = transforms.ToPILImage()
    model.eval()
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            t   = to_t(img).unsqueeze(0).to(DEVICE)
            enh, _ = model(t)
            out = to_p(enh.squeeze(0).clamp(0, 1).cpu())
            out = out.resize(img.size, Image.LANCZOS)
            out.save(os.path.join(out_dir, os.path.basename(p)))

    print(f"  {split}: {len(paths)} images enhanced → {out_dir}")

    # copy labels unchanged (bounding boxes don't change)
    lbl_src = os.path.join(DATASET_ROOT, split, "labels")
    lbl_dst = os.path.join(ENHANCED_ROOT, split, "labels")
    if os.path.exists(lbl_src):
        shutil.copytree(lbl_src, lbl_dst, dirs_exist_ok=True)
        print(f"  {split}: labels copied → {lbl_dst}")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Train Zero-DCE
    print("\n=== Step 1: Training Zero-DCE ===")
    dirs = [os.path.join(DATASET_ROOT, s, "images")
            for s in ["train", "valid"]
            if os.path.exists(os.path.join(DATASET_ROOT, s, "images"))]
    ds    = ImageDataset(dirs)
    dl    = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = ZeroDCE().to(DEVICE)
    crit  = ZeroDCELoss()
    opt   = optim.Adam(model.parameters(), lr=LR)
    train_zerodce(model, dl, crit, opt)
    torch.save(model.state_dict(), "zerodce_weights.pt")
    print("Zero-DCE weights saved → zerodce_weights.pt")

    # Enhance all splits
    print("\n=== Step 2: Enhancing Dataset Images ===")
    for split in ["train", "valid", "test"]:
        print(f"\n  Processing: {split}")
        enhance_and_save(model, split)

    # Write data.yaml for enhanced dataset
    print("\n=== Step 3: Writing data.yaml ===")
    src_yaml = os.path.join(DATASET_ROOT, "data.yaml")
    dst_yaml = os.path.join(ENHANCED_ROOT, "data.yaml")
    with open(src_yaml) as f:
        content = f.read()
    with open(dst_yaml, "w") as f:
        f.write(content.replace("Annotations-1", "Annotations-1-enhanced"))
    print(f"data.yaml → {dst_yaml}")

    print("\n✓ Done! Now run: python train_yolov11_zerodce.py")