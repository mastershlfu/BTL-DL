import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
import random

# --- CẤU HÌNH ---
CHECKPOINT_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/faster_rcnn_logs/fasterrcnn_epoch_7.pth"
IMAGE_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/gai_thread.jpeg"
OUTPUT_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/outputs/inference_fastRCNN/eval_threads_result.jpg"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- 1. DANH SÁCH TÊN CLASS CỦA COCO (91 Classes) ---
# Đây là mapping chuẩn của PyTorch Faster R-CNN pre-trained
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- 2. TẠO MÀU NGẪU NHIÊN CHO TỪNG CLASS ---
# Sinh ra 91 màu khác nhau (mỗi lần chạy sẽ giống nhau nhờ seed)
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    # 1. Load Model
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    model = get_model(num_classes=91)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    # 2. Load Ảnh
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print(f"❌ Không tìm thấy ảnh tại: {IMAGE_PATH}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Resize về 800x800 để đưa vào model (giống lúc train/val thường dùng)
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor()
    ])
    img_tensor = transform(img_pil).to(DEVICE)

    # 3. Dự đoán
    print("🔍 Running inference...")
    with torch.no_grad():
        predictions = model([img_tensor])

    # 4. Xử lý kết quả
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy() # Lấy thêm labels ID

    # Tính tỉ lệ scale để vẽ lại lên ảnh gốc
    orig_h, orig_w = img_bgr.shape[:2]
    scale_x = orig_w / 800
    scale_y = orig_h / 800

    print(f"Found {len(boxes)} raw boxes")
    
    output_img = img_bgr.copy()
    count = 0
    
    for i, box in enumerate(boxes):
        score = scores[i]
        if score > 0.5: # Ngưỡng tin cậy
            count += 1
            label_id = labels[i]
            
            # Lấy tên class và màu sắc tương ứng
            # Dùng try-except để tránh lỗi nếu model dự đoán ra ID lạ (dù hiếm)
            try:
                class_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
            except IndexError:
                class_name = f"Class {label_id}"
            
            color = COLORS[label_id] # Màu theo ID class (BGR cho OpenCV)
            
            # Scale box về ảnh gốc
            x1, y1, x2, y2 = box
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Vẽ Box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ nền chữ (để chữ dễ đọc hơn)
            text_label = f"{class_name}: {score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1) # Box đặc (-1)
            
            # Viết chữ (Màu trắng hoặc đen tùy nền, ở đây để trắng cho nổi trên nền màu)
            cv2.putText(output_img, text_label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print(f"Saved visualization with {count} objects to '{OUTPUT_PATH}'")
    cv2.imwrite(OUTPUT_PATH, output_img)

if __name__ == "__main__":
    main()