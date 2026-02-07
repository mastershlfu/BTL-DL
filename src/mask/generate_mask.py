import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import random

# = CẤU HÌNH ĐƯỜNG DẪN =
# Đường dẫn chứa ảnh gốc (Dựa trên cấu trúc của bạn)
INPUT_DIR = '/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/val_large' 
# Đường dẫn sẽ lưu mask
OUTPUT_DIR = '/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places2/masks' 

# Cấu hình Mask
IMG_SIZE = 256       # Kích thước ảnh (theo config RePaint của bạn là 256)
MIN_STROKES = 1      # Số nét vẽ tối thiểu
MAX_STROKES = 6      # Số nét vẽ tối đa
MIN_WIDTH = 10       # Độ dày nét vẽ tối thiểu
MAX_WIDTH = 40       # Độ dày nét vẽ tối đa

def create_random_mask(height, width):
    """
    Tạo mask đen trắng với các nét vẽ ngẫu nhiên
    - Nền TRẮNG (255): Giữ nguyên
    - Nét vẽ ĐEN (0): Cần xóa (Inpaint)
    """
    
    mask = np.ones((height, width), np.uint8) * 255
    
    num_strokes = random.randint(MIN_STROKES, MAX_STROKES)
    
    for _ in range(num_strokes):
        # Chọn 2 điểm ngẫu nhiên
        x1, x2 = random.randint(0, width), random.randint(0, width)
        y1, y2 = random.randint(0, height), random.randint(0, height)
        
        thickness = random.randint(MIN_WIDTH, MAX_WIDTH)
        
        # Vẽ đường trắng lên nền đen
        cv2.line(mask, (x1, y1), (x2, y2), (0), thickness)
        
        # Đôi khi vẽ thêm hình tròn ở đầu mút cho tự nhiên
        if random.random() > 0.5:
            cv2.circle(mask, (x1, y1), thickness // 2, (0), -1)
        if random.random() > 0.5:
            cv2.circle(mask, (x2, y2), thickness // 2, (0), -1)
            
    return mask

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {OUTPUT_DIR}")

    # Lấy danh sách ảnh (hỗ trợ jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        # Thử tìm cả trong folder con nếu file zip được giải nén thành folder con
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, '*', ext)))

    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu tạo mask...")

    for path in tqdm(image_paths):
        # Lấy tên file
        filename = os.path.basename(path)
        
        # Đọc ảnh gốc để lấy kích thước thật (hoặc fix cứng 256x256)
        # img = cv2.imread(path)
        # h, w = img.shape[:2]
        
        # Vì config RePaint của bạn set image_size: 256, ta nên tạo mask 256x256
        h, w = IMG_SIZE, IMG_SIZE
        
        # Tạo mask
        mask = create_random_mask(h, w)
        
        # Lưu mask (Mask phải cùng tên với ảnh gốc)
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # RePaint thường yêu cầu: Vùng xóa là TRẮNG (255), vùng giữ là ĐEN (0)
        # Hoặc ngược lại tùy config. Mặc định thường là Mask = 255 (trắng).
        cv2.imwrite(save_path, mask)

    print("Hoàn tất! Kiểm tra folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()