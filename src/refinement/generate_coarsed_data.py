import sys
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import glob
import argparse

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import RePaint Wrapper (đã viết ở bước trước)
from src.inpaint.repaint import RePaintInpainter

# Config
REPAINT_MODEL_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/RePaint/log/places2_256_thin/model.pt"
INPUT_DIR  = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/val_large"
OUTPUT_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/coarse_images"
IMG_SIZE = 256 # RePaint Places2 model chỉ chạy chuẩn ở 256x256

def random_mask(height, width):
    """Tạo mask ngẫu nhiên (vuông + nét vẽ)"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Random Rectangles
    for _ in range(np.random.randint(1, 4)):
        x1 = np.random.randint(0, width - 50)
        y1 = np.random.randint(0, height - 50)
        w = np.random.randint(30, 100)
        h = np.random.randint(30, 100)
        cv2.rectangle(mask, (x1, y1), (x1+w, y1+h), 255, -1)
        
    # Random Strokes
    for _ in range(np.random.randint(1, 4)):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        thickness = np.random.randint(10, 30)
        for _ in range(5):
            x2 = x1 + np.random.randint(-40, 40)
            y2 = y1 + np.random.randint(-40, 40)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
            x1, y1 = x2, y2
    return mask

def main():
    # 1. Setup Folders
    os.makedirs(os.path.join(OUTPUT_DIR, "gt"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "mask"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "coarse"), exist_ok=True)

    # 2. Load RePaint
    print("⏳ Loading RePaint...")
    painter = RePaintInpainter(REPAINT_MODEL_PATH, device="cuda")

    # 3. Get Image List
    img_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    img_paths = img_paths[:5000] # Demo thì lấy 5000 ảnh thôi, train thật thì bỏ dòng này
    
    print(f"🚀 Processing {len(img_paths)} images...")

    for i, path in enumerate(tqdm(img_paths)):
        fname = os.path.basename(path)
        
        # A. Đọc ảnh & Resize
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # B. Tạo Mask
        mask = random_mask(IMG_SIZE, IMG_SIZE)
        
        # C. Chạy RePaint (Tạo ảnh Coarse)
        # Hàm inpaint này bạn đã có ở wrapper trước
        try:
            # RePaintWrapper nhận ảnh BGR hoặc RGB đều được (nó tự convert)
            coarse_img = painter.inpaint(img, mask)
        except Exception as e:
            print(f"Error on {fname}: {e}")
            continue

        # D. Lưu dữ liệu
        cv2.imwrite(os.path.join(OUTPUT_DIR, "gt", fname), img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "mask", fname), mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "coarse", fname), coarse_img)

if __name__ == "__main__":
    main()