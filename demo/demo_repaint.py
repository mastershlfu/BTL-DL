import gradio as gr
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Thêm đường dẫn root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.fasterRCNN_SAM_LaMa import ObjectRemovalSystem
from src.refinement.network import RefinementUNet

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RCNN_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/pipeline/models/faster_rcnn_logs/fasterrcnn_epoch_7.pth"
SAM_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/sam_vit_h_4b8939.pth"
LAMA_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/lama/big-lama.pt"
REFINE_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/src/refinement/models/refinement_logs/refine_epoch_5.pth"
REPAINT_CONF_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/models/RePaint/confs/test_p256_thin.yml"

# --- INIT SYSTEM ---
# Pipeline này đã được bạn cập nhật để dùng RePaint tạo Coarse Image
pipeline = ObjectRemovalSystem(RCNN_PATH, SAM_PATH, REPAINT_CONF_PATH, device=DEVICE)

# Khởi tạo mạng Refinement
refine_net = RefinementUNet(in_channels=4, out_channels=3).to(DEVICE)
if os.path.exists(REFINE_PATH):
    print(f"✅ Loading Refinement weights from {REFINE_PATH}")
    refine_net.load_state_dict(torch.load(REFINE_PATH, map_location=DEVICE))
refine_net.eval()

def apply_gaussian_blur(mask_tensor, kernel_size=7, sigma=2.0):
    if kernel_size % 2 == 0: kernel_size += 1
    x_coord = torch.arange(kernel_size).float()
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*np.pi*variance)) * torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(mask_tensor.device)
    pad = kernel_size // 2
    blurred_mask = F.conv2d(mask_tensor, gaussian_kernel, padding=pad)
    return blurred_mask

def run_refinement(coarse_img_rgb, mask_255, original_img_rgb):
    """
    Hàm Inference khớp 100% với logic Training
    """
    h, w = original_img_rgb.shape[:2]
    
    # 1. Chuyển đổi sang Tensor [1, C, H, W]
    coarse_t = torch.from_numpy(coarse_img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    mask_t = torch.from_numpy(mask_255).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
    orig_t = torch.from_numpy(original_img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

    # 2. Logic Mask Inversion (Giống train)
    mask_for_net = 1.0 - mask_t 
    
    # 3. Resize về size training (256x256)
    coarse_in = F.interpolate(coarse_t, size=(256, 256), mode='bilinear')
    mask_in = F.interpolate(mask_for_net, size=(256, 256), mode='nearest')

    with torch.no_grad():
        refined_t = refine_net(coarse_in, mask_in)
        # Resize ngược về size gốc
        refined_t = F.interpolate(refined_t, size=(h, w), mode='bilinear')

    # 4. Blending với Gaussian Blur
    mask_blur = apply_gaussian_blur(mask_t, kernel_size=7, sigma=2.0)
    
    # Pasting Logic: Giữ nguyên vùng ngoài từ ảnh gốc xịn nhất
    final_t = orig_t * (1.0 - mask_blur) + refined_t * mask_blur

    # 5. Convert back to Numpy
    final_res = (final_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return final_res

def load_image_from_gradio(img_data):
    if img_data is None: return None
    if isinstance(img_data, str):
        img = cv2.imread(img_data)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img_data, np.ndarray):
        # Ép về 3 kênh (RGB)
        if img_data.shape[-1] == 4:
            return cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        return img_data
    return None

def gradio_process(input_dict, text_prompt):
    if input_dict is None or input_dict.get("background") is None:
        return None, None, None, "⚠️ Vui lòng upload ảnh!"

    # 1. Load background (Ảnh gốc)
    image_rgb = load_image_from_gradio(input_dict["background"])
    
    # 2. Xử lý Mask vẽ tay (Layers)
    mask_manual = None
    layers = input_dict.get("layers", [])
    if layers and len(layers) > 0:
        draw_layer = load_image_from_gradio(layers[0])
        if draw_layer is not None and np.any(draw_layer):
            # Nếu là RGBA lấy kênh alpha, nếu không convert xám
            if draw_layer.shape[-1] == 4:
                mask_manual = draw_layer[:, :, 3]
            else:
                mask_manual = cv2.cvtColor(draw_layer, cv2.COLOR_RGB2GRAY)
            mask_manual[mask_manual > 0] = 255

    # 3. Chạy Pipeline (Faster R-CNN -> SAM -> RePaint)
    try:
        # Pipeline sẽ trả về mask_res (255=hole) và coarse_res (RePaint output)
        mask_res, coarse_res, status = pipeline.process(image_rgb, mask_manual, text_prompt)
        
        if coarse_res is None:
            return None, None, None, status
        
        # 4. Chạy CNN Refinement
        final_refined = run_refinement(coarse_res, mask_res, image_rgb)
            
        return mask_res, coarse_res, final_refined, status
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"❌ Lỗi hệ thống: {str(e)}"

# --- UI LAYOUT ---
with gr.Blocks(title="Object Removal Pro") as demo:
    gr.Markdown("# 🪄 Deep Learning Object Removal")
    gr.Markdown("Pipeline: Faster R-CNN + SAM -> RePaint (Coarse) -> CNN (Refinement)")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_editor = gr.ImageEditor(
                label="Input Image & Brush",
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed", default_size=20),
                interactive=True,
                sources=["upload"],
                type="numpy",
                transforms=[]
            )
            text_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. person, car, dog...")
            btn_run = gr.Button("🚀 Run Removal", variant="primary")
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("✨ Refined Result"):
                    output_final = gr.Image(label="Final Output")
                with gr.Tab("🖼️ Coarse (RePaint)"):
                    output_coarse = gr.Image(label="Coarse Image")
                with gr.Tab("🎭 Mask"):
                    output_mask = gr.Image(label="Mask Result", image_mode="L")
            
            status_text = gr.Label(label="Status")

    btn_run.click(
        fn=gradio_process,
        inputs=[input_editor, text_prompt],
        outputs=[output_mask, output_coarse, output_final, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=True, show_api=False)