import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import cv2
import numpy as np
from src.inpaint.repaint import RePaintInpainter

# Đường dẫn model bạn tải ở Bước 3
MODEL_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/repaint/256x256_diffusion_uncond.pt"

def main():
    # 1. Khởi tạo
    painter = RePaintInpainter(MODEL_PATH)
    
    # 2. Tạo ảnh giả
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (300, 300), 255, -1) # Mask hình vuông trắng
    
    # 3. Inpaint
    print(" Start Inpainting...")
    output = painter.inpaint(img, mask)
    print(" Done! Output shape:", output.shape)
    
    cv2.imwrite("/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/coarse_images", output)

if __name__ == "__main__":
    main()