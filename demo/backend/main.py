from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lấy dir ảnh để đồng bộ cho inference
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(project_root, "data", "LaMa_test_images", "images")

class Box(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class SubmitPayload(BaseModel):
    image_name: str
    boxes: list[Box]

@app.post("/submit_boxes")
def submit_boxes(data: SubmitPayload):
    # ghep path
    full_path = os.path.abspath(os.path.join(IMAGE_DIR, data.image_name))
    
    # kiem tra path
    file_exists = os.path.exists(full_path)
    output_txt_path = os.path.join(project_root, "img_path.txt")
    # --- PIPELINE INFERENCE Ở ĐÂY ---
    # Ví dụ: 
    # if file_exists:
    #     prediction = model.predict(full_path, data.boxes)
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_path)
        print(f"✅ Đã ghi đường dẫn vào: {output_txt_path}")
    except Exception as e:
        print(f"❌ Lỗi ghi file: {e}")

    print(f"Dữ liệu nhận được cho ảnh: {full_path}")
    print(f"boxes: {data.boxes}")
    print(f"Full path: {full_path}")
    
    return {
        "status": "success",
        "num_boxes": len(data.boxes),
        "absolute_path": full_path,
        "file_exists": file_exists,
        "message": f"Dữ liệu đã sẵn sàng cho pipeline tại {full_path}"
    }