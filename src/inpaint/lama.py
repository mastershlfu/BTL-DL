import torch
import numpy as np
import cv2
import torch.nn.functional as F

class LaMaInpainter:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"🎨 Loading LaMa model from {model_path}...")
        # Load model TorchScript (cách chuẩn và gọn nhất của LaMa)
        try:
            self.model = torch.jit.load(model_path, map_location=device)
            self.model.eval()
            self.model.to(device)
        except Exception as e:
            print(f"❌ Error loading LaMa: {e}")
            raise e

    def inpaint(self, image_np, mask_np):
        """
        image_np: Ảnh RGB (H, W, 3) hoặc RGBA (H, W, 4) - uint8 0..255
        mask_np: Ảnh Mask (H, W) hoặc (H, W, 1) - uint8 0..255
        """
        # 1. Đảm bảo ảnh chỉ có 3 kênh RGB (Loại bỏ Alpha nếu có)
        if image_np.ndim == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
            
        # 2. Pre-process Image & Mask
        # Chuyển về float 0-1 và format (C, H, W)
        img = torch.from_numpy(image_np).permute(2, 0, 1).float().div(255.)
        
        # Xử lý mask: Nếu mask 2D thì thêm dim
        if mask_np.ndim == 2:
            mask_np = mask_np[:, :, np.newaxis]
        mask = torch.from_numpy(mask_np).permute(2, 0, 1).float().div(255.)
        
        # [FIX LỖI 5 KÊNH] Threshold mask về 0 và 1 tuyệt đối
        mask = (mask > 0.5).float()

        # 3. Thêm batch dimension
        img = img.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        # 4. Padding cho chia hết cho 8
        h, w = img.shape[2], img.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='reflect')

        # 5. Inference
        with torch.no_grad():
            # LaMa TorchScript (big-lama.pt) yêu cầu input là (img, mask) riêng biệt
            # Model sẽ tự concat bên trong thành 4 kênh. 
            # Đảm bảo img là 3 kênh, mask là 1 kênh.
            output = self.model(img, mask)

        # 6. Post-process
        # Cắt bỏ phần padding
        output = output[:, :, :h, :w]
        
        # Chuyển về numpy (H, W, C)
        output = output[0].permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        return output