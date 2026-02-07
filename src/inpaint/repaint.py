import os
import sys
import torch as th
import cv2
import numpy as np

# 1. SETUP ĐƯỜNG DẪN REPAINT
REPAINT_ROOT = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/RePaint"
if REPAINT_ROOT not in sys.path:
    sys.path.insert(0, REPAINT_ROOT)

import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    create_model_and_diffusion, 
    select_args, 
    model_and_diffusion_defaults,
    NUM_CLASSES
)

class RePaintInpainter:
    def __init__(self, config_path, device="cuda"):
        self.device = device
        print(f"  📖 Loading RePaint Config: {config_path}")
        
        # Load config chuẩn từ RePaint
        self.conf = conf_mgt.conf_base.Default_Conf()
        self.conf.update(yamlread(config_path))
        
        # Khởi tạo Model và Diffusion
        model_args = select_args(self.conf, model_and_diffusion_defaults().keys())
        self.model, self.diffusion = create_model_and_diffusion(
            **model_args,
            conf=self.conf
        )
        
        # Load Weights (Sử dụng đường dẫn trong file yml)
        model_full_path = self.conf.model_path
        print(f"  📦 Loading Weights: {model_full_path}")
        self.model.load_state_dict(
            dist_util.load_state_dict(model_full_path, map_location="cpu")
        )
        
        self.model.to(self.device)
        if self.conf.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()
        print("  ✅ RePaint Loaded Successfully!")

    # def inpaint(self, image_np, mask_np):
    #     """
    #     image_np: RGB (H, W, 3), 0-255
    #     mask_np: Gray (H, W), 255 là lỗ, 0 là nền
    #     """
    #     h_orig, w_orig = image_np.shape[:2]
        
    #     # 1. Preprocess (Bắt buộc 256x256 cho model Places2)
    #     img_res = cv2.resize(image_np, (256, 256))
    #     mask_res = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_NEAREST)

    #     # Chuyển sang tensor [-1, 1] cho Model
    #     gt_tensor = th.from_numpy(img_res).float() / 127.5 - 1.0
    #     gt_tensor = gt_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

    #     # Chuyển mask sang 0-1 (1 là vùng giữ, 0 là vùng lỗ - theo chuẩn RePaint)
    #     # Lưu ý: RePaint dùng gt_keep_mask (1 = giữ, 0 = xóa)
    #     gt_keep_mask = th.from_numpy(mask_res).float() / 255.0
    #     gt_keep_mask = (gt_keep_mask < 0.5).float() # Đảo ngược: 255 (lỗ) -> 0, 0 (nền) -> 1
    #     gt_keep_mask = gt_keep_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        
    #     # 2. Setup Model Function (Giống test.py)
    #     def model_fn(x, t, y=None, gt=None, **kwargs):
    #         return self.model(x, t, y if self.conf.class_cond else None, gt=gt)

    #     # 3. Setup Model Kwargs
    #     model_kwargs = {
    #         "gt": gt_tensor,
    #         "gt_keep_mask": gt_keep_mask
    #     }
        
    #     # Xử lý Class Condition
    #     batch_size = 1
    #     if self.conf.cond_y is not None:
    #         classes = th.ones(batch_size, dtype=th.long, device=self.device)
    #         model_kwargs["y"] = classes * self.conf.cond_y
    #     else:
    #         classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=self.device)
    #         model_kwargs["y"] = classes

    #     # 4. RUN SAMPLING LOOP (Sử dụng hàm của thư viện để tránh lỗi conf)
    #     print("    Sampling with RePaint (Diffusion)...")
    #     with th.no_grad():
    #         sample_fn = self.diffusion.p_sample_loop if not self.conf.use_ddim else self.diffusion.ddim_sample_loop
            
    #         result = sample_fn(
    #             model_fn,
    #             (batch_size, 3, 256, 256),
    #             clip_denoised=self.conf.clip_denoised,
    #             model_kwargs=model_kwargs,
    #             device=self.device,
    #             progress=True, # Hiện thanh tiến trình trong terminal
    #             return_all=False,
    #             conf=self.conf # QUAN TRỌNG: Phải truyền conf vào đây
    #         )

    #     # 5. Post-process
    #     # Đưa từ [-1, 1] về [0, 255]
        
    #     out_img = ((result + 1) * 127.5).clamp(0, 255).to(th.uint8)
    #     out_img = out_img.permute(0, 2, 3, 1).cpu().numpy()[0]
        
    #     # Resize lại kích thước ban đầu
    #     final_output = cv2.resize(out_img, (w_orig, h_orig))
        
    #     return final_output
    
    def inpaint(self, image_np, mask_np):
        """
        image_np: RGB uint8 (H, W, 3 hoặc 4), [0,255]
        mask_np : uint8 (H, W), 255 = hole, 0 = keep
        """
        h_orig, w_orig = image_np.shape[:2]

        # --- BƯỚC 1: ĐẢM BẢO ẢNH LÀ 3 KÊNH (RGB) ---
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]
        
        # --------------------------------------------------
        # 2. Resize về 256x256
        # --------------------------------------------------
        img_res = cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_NEAREST)

        # --------------------------------------------------
        # 3. Normalize image sang Tensor [-1, 1] (3 channels)
        # --------------------------------------------------
        img_res = img_res.astype(np.float32) / 127.5 - 1.0
        gt_tensor = th.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # --------------------------------------------------
        # 4. gt_keep_mask: 1 = keep, 0 = hole (1 channel)
        # --------------------------------------------------
        # Trong mask_np: 255 là lỗ (hole). RePaint cần lỗ = 0.
        keep_mask = (mask_res < 128).astype(np.float32)
        gt_keep_mask = th.from_numpy(keep_mask).unsqueeze(0).unsqueeze(0).to(self.device)

        # --------------------------------------------------
        # 5. Model function & Kwargs (Bám sát test.py)
        # --------------------------------------------------
        def model_fn(x, t, y=None, gt=None, **kwargs):
            return self.model(x, t, y if self.conf.class_cond else None, gt=gt)

        model_kwargs = {
            "gt": gt_tensor,        # Đảm bảo 3 kênh
            "gt_keep_mask": gt_keep_mask # Đảm bảo 1 kênh
        }

        batch_size = 1
        if self.conf.cond_y is not None:
            model_kwargs["y"] = th.tensor([self.conf.cond_y], device=self.device, dtype=th.long)
        else:
            model_kwargs["y"] = th.randint(0, NUM_CLASSES, (batch_size,), device=self.device)

        # --------------------------------------------------
        # 6. Sampling (3 channels)
        # --------------------------------------------------
        print(f"    [RePaint] Sampling start... (GT shape: {gt_tensor.shape})")
        with th.no_grad():
            sample_fn = (
                self.diffusion.p_sample_loop
                if not self.conf.use_ddim
                else self.diffusion.ddim_sample_loop
            )

            result = sample_fn(
                model_fn,
                (batch_size, 3, 256, 256), # Bắt buộc là 3 kênh cho Places2
                clip_denoised=self.conf.clip_denoised,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=True,
                return_all=False,
                conf=self.conf,
            )

        # --------------------------------------------------
        # 7. Post-process
        # --------------------------------------------------
        out = ((result + 1) * 127.5).clamp(0, 255).to(th.uint8)
        out = out.permute(0, 2, 3, 1).cpu().numpy()[0]

        # Resize ngược về kích thước ban đầu của User
        out = cv2.resize(out, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return out
