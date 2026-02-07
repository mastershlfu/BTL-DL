import os
import sys
import cv2
import glob
import numpy as np
from tqdm import tqdm
import random
import torch
import argparse

# ---------------- CONFIG ----------------
INPUT_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/val_large"
MASK_DIR  = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/masks"
COARSE_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/coarse_images"

IMG_SIZE = 256
DEVICE = "cuda"

SAM_CKPT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
LAMA_CKPT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/lama/big-lama.pt"

# ----------------------------------------

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(COARSE_DIR, exist_ok=True)


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# -------- LOAD SAM --------
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    min_mask_region_area=500
)

# -------- LOAD LAMA --------
from src.inpaint.lama import LaMaInpainter
lama = LaMaInpainter(LAMA_CKPT, device=DEVICE)

# ----------------------------------------

def generate_sam_mask(image_np, max_objects=2):
    """
    image_np: RGB uint8 (H,W,3)
    return: mask uint8 (H,W) | 255 = hole, 0 = keep
    """
    H, W, _ = image_np.shape
    masks = mask_generator.generate(image_np)

    if len(masks) == 0:
        return np.zeros((H, W), np.uint8)

    # sort theo area lớn → nhỏ
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    num_obj = random.randint(1, min(max_objects, len(masks)))
    selected = random.sample(masks[:10], num_obj)

    final_mask = np.zeros((H, W), dtype=np.uint8)
    for m in selected:
        final_mask[m["segmentation"]] = 255

    return final_mask


def main():
    args = parse_args()

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, "*", ext)))

    image_paths = sorted(image_paths)
    total = len(image_paths)

    start = args.start
    end = args.end if args.end > 0 else total
    end = min(end, total)

    print(f"🔍 Total images: {total}")
    print(f"🚀 Processing range: [{start}:{end})")

    for idx in tqdm(range(start, end)):
        img_path = image_paths[idx]
        filename = os.path.basename(img_path)

        coarse_path = os.path.join(COARSE_DIR, filename)
        mask_path   = os.path.join(MASK_DIR, filename)

        if os.path.exists(mask_path) and os.path.exists(coarse_path):
            continue
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        # --- SAM MASK ---
        mask = generate_sam_mask(img_rgb)
        cv2.imwrite(os.path.join(MASK_DIR, filename), mask)

        # --- LAMA COARSE ---
        coarse = lama.inpaint(img_rgb, mask)
        coarse_bgr = cv2.cvtColor(coarse, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(COARSE_DIR, filename), coarse_bgr)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="start index")
    parser.add_argument("--end", type=int, default=-1, help="end index (exclusive)")
    return parser.parse_args()

if __name__ == "__main__":
    main()
