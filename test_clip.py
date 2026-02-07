import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 1. Load và tiền xử lý ảnh
image_path = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Untitled.jpeg"
image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

# 2. Tokenize văn bản
text_inputs = clip.tokenize([
    "a photo of a person",
    "a photo of a car",
    "a photo of a tree"
]).to(device)

with torch.no_grad():
    # 3. Trích xuất đặc trưng (Features)
    img_feat = model.encode_image(image)
    txt_feat = model.encode_text(text_inputs)

    # 4. Chuẩn hóa vector (Quan trọng để tính cosine similarity)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    # 5. Lấy hệ số logit_scale có sẵn trong model (thường là 100)
    logit_scale = model.logit_scale.exp()
    
    # 6. Tính toán logits và xác suất bằng Softmax
    logits_per_image = logit_scale * img_feat @ txt_feat.T
    probs = logits_per_image.softmax(dim=-1)

print("CLIP Probabilities:", probs.cpu().numpy())