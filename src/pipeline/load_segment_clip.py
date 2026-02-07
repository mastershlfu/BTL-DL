import torch
import clip
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load SAM
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 2. Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)