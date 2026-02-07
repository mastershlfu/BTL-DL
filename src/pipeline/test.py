import torch
import numpy as np
import cv2
import os
from utils.get_img import get_image_path_from_txt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# --- CẤU HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
IMAGE_PATH = get_image_path_from_txt()

INPUT_BOX = np.array([211, 526, 581, 1339])

# def main():
#     print(f"1. Đang đọc ảnh từ file: {IMAGE_PATH}")
#     if not os.path.exists(IMAGE_PATH):
#         print(f"❌ Lỗi: Không tìm thấy file ảnh tại {IMAGE_PATH}")
#         return

#     image_bgr = cv2.imread(IMAGE_PATH)
#     if image_bgr is None:
#         print("❌ Lỗi: Không thể đọc file ảnh")
#         return

#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#     # Tạo folder output
#     output_dir = "../../outputs/mask_all"
#     os.makedirs(output_dir, exist_ok=True)
#     cv2.imwrite(f"{output_dir}/input.jpg", image_bgr)
    
#     print("2. Đang load SAM (Không cần CLIP)...")
#     sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
#     sam.to(device=DEVICE)
    
#     # SamAutomaticMaskGenerator chuyên dùng để quét toàn bộ ảnh
#     # Bạn có thể chỉnh points_per_side=32 để quét kỹ hơn, hoặc 16 để nhanh hơn
#     mask_generator = SamAutomaticMaskGenerator(sam)

#     print("3. SAM đang quét toàn bộ ảnh...")
#     all_masks = mask_generator.generate(image_rgb)
#     print(f"   -> Tìm thấy {len(all_masks)} vùng (segments).")

#     # --- GỘP TẤT CẢ MASK LẠI ---
#     h, w, _ = image_rgb.shape
#     combined_mask = np.zeros((h, w), dtype=bool)

#     # Sắp xếp mask theo diện tích (để vẽ cái to trước cái nhỏ sau nếu cần visualize)
#     all_masks = sorted(all_masks, key=lambda x: x['area'], reverse=True)

#     for ann in all_masks:
#         m = ann['segmentation']
#         combined_mask = np.logical_or(combined_mask, m)

#     # --- LƯU KẾT QUẢ ---
#     print("4. Đang lưu kết quả...")
    
#     # 1. Lưu Binary Mask (Trắng đen) - Cái này để đưa vào LaMa
#     mask_uint8 = (combined_mask * 255).astype(np.uint8)
#     # Nở mask ra chút xíu để xóa cho sạch
#     kernel = np.ones((5, 5), np.uint8) 
#     dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    
#     cv2.imwrite(f"{output_dir}/mask_all.png", dilated_mask)

#     # 2. Tạo ảnh Visualization (Các mảng màu ngẫu nhiên) để dễ nhìn xem nó cắt những gì
#     # Tạo một ảnh đen để vẽ màu lên
#     vis_image = np.zeros_like(image_bgr)
    
#     for ann in all_masks:
#         m = ann['segmentation']
#         # Tạo màu ngẫu nhiên cho mỗi segment
#         color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
#         vis_image[m] = color

#     # Trộn ảnh màu mask với ảnh gốc (độ trong suốt 0.5)
#     overlay = cv2.addWeighted(image_bgr, 0.5, vis_image, 0.5, 0)
#     cv2.imwrite(f"{output_dir}/preview_all.jpg", overlay)

#     print(f"✅ ĐÃ XONG! Kiểm tra folder: {output_dir}")
#     print("   - mask_all.png: Mask tổng hợp (Input cho LaMa)")
#     print("   - preview_all.jpg: Ảnh xem trước các vùng đã cắt")

# if __name__ == "__main__":
#     main()

def main():
    print(f"1. Đang đọc ảnh từ file: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Lỗi: Không tìm thấy file ảnh tại {IMAGE_PATH}")
        return

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    output_dir = "../../outputs/mask_box"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/input.jpg", image_bgr)
    
    print("2. Đang load SAM Predictor...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    
    predictor = SamPredictor(sam)

    print("3. Đang mã hóa ảnh (Image Embedding)...")
    predictor.set_image(image_rgb)

    print(f"4. Đang tạo mask từ Box: {INPUT_BOX}...")
    
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=INPUT_BOX[None, :],
        multimask_output=True   
    )

    best_idx = np.argmax(scores)
    final_mask = masks[best_idx] 
    print(f"   -> Đã chọn mask có score cao nhất: {scores[best_idx]:.4f}")

    print("5. Đang lưu kết quả...")
    
    mask_uint8 = (final_mask * 255).astype(np.uint8)
    
    kernel = np.ones((5, 5), np.uint8) 
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    cv2.imwrite(f"{output_dir}/intput_mask.png", dilated_mask)

    vis_image = image_bgr.copy()
    
    color_mask = np.zeros_like(image_bgr)
    color_mask[final_mask] = [0, 255, 0] 
    vis_image = cv2.addWeighted(vis_image, 1.0, color_mask, 0.5, 0)

    x1, y1, x2, y2 = INPUT_BOX
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imwrite(f"{output_dir}/preview_box.jpg", vis_image)

    print(f"✅ ĐÃ XONG! Kiểm tra folder: {output_dir}")

if __name__ == "__main__":
    main()