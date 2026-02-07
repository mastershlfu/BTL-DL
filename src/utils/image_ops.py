import cv2
import numpy as np
from skimage.exposure import match_histograms

def refine_mask_iopaint(mask_uint8, dilate_k=15, blur_k=21):
    """
    (A) Mask Refine: Mở rộng và làm mềm biên
    mask_uint8: 255 là lỗ hổng, 0 là nền
    """
    # 1. Dilate (Nở mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # 2. Smooth (Làm mờ biên)
    mask_f = dilated.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(mask_f, (blur_k, blur_k), 0)
    return blurred # Trả về Alpha Map [0, 1]

def apply_post_process(original_rgb, inpainted_rgb, mask_alpha):
    """
    (C) Post-process: Histogram Match + Alpha Blend
    """
    # 1. Histogram Matching: Ép vùng vẽ mới theo tone màu gốc
    matched = match_histograms(inpainted_rgb, original_rgb, channel_axis=-1)
    
    # 2. Edge Feathering (Alpha Blending)
    alpha = np.expand_dims(mask_alpha, axis=-1)
    final = (1.0 - alpha) * original_rgb + alpha * matched
    return final.clip(0, 255).astype(np.uint8)