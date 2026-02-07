import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG19Features(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Chúng ta lấy feature ở các tầng khác nhau để bắt Texture (nông) và Structure (sâu)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(2): self.slice1.add_module(str(x), vgg[x])   # relu1_1
        for x in range(2, 7): self.slice2.add_module(str(x), vgg[x]) # relu2_1
        for x in range(7, 12): self.slice3.add_module(str(x), vgg[x]) # relu3_1
        for x in range(12, 21): self.slice4.add_module(str(x), vgg[x]) # relu4_1
        
        # Freeze VGG (Không train)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4

class InpaintingLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.vgg = VGG19Features().to(device).eval()
        self.l1 = nn.L1Loss()
        
        # Trọng số cho các loại loss (Hyperparameters quan trọng)
        self.w_l1 = 1.0
        self.w_perc = 0.1
        self.w_style = 50.0  # Style loss thường có giá trị rất nhỏ nên cần weight to
        self.w_grad = 0.3

    def gram_matrix(self, feat):
        # Tính Gram Matrix cho Style Loss
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
        return gram
    
    def resize_mask(self, mask, target_feat):
        """
        mask: [B, 1, H, W]
        target_feat: [B, C, h, w]
        """
        return F.interpolate(
            mask,
            size=target_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    def gradient_loss(self, pred, target):
        # Tính đạo hàm theo hướng x và y (Sobel đơn giản)
        # So sánh độ chênh lệch pixel giữa các điểm lân cận
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        loss = torch.mean(torch.abs(pred_dx - target_dx)) + \
               torch.mean(torch.abs(pred_dy - target_dy))
        return loss

    def normalize_vgg(self, x):
        # Giả sử x đang ở dải [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
            return (x - mean) / std

    def forward(self, refined_img, gt_img, mask):
        # --- 1. L1 Loss (Pixel Consistency) ---
        # Chỉ tính trong vùng mask (vùng bị xóa)
        loss_l1 = torch.mean(
            torch.abs(refined_img - gt_img) * mask
        )

        # Chuẩn bị input cho VGG (Normalize theo chuẩn ImageNet nếu cần, ở đây giả sử input đã chuẩn)
        # Trích xuất features
        feat_pred = self.vgg(self.normalize_vgg(refined_img)) # Trả về list 4 tensors
        feat_gt = self.vgg(self.normalize_vgg(gt_img))

        # --- 2. Perceptual Loss (Structure) ---
        loss_perc = 0
        for fp, fg in zip(feat_pred, feat_gt):
            mask_resized = self.resize_mask(mask, fp)
            loss_perc += torch.mean(torch.abs(fp - fg) * mask_resized)
        # loss_perc = 0
        # for i in range(len(feat_pred)):
        #     loss_perc += self.l1(feat_pred[i], feat_gt[i])

        # --- 3. Style Loss (Texture) ---
        loss_style = 0
        for fp, fg in zip(feat_pred[:2], feat_gt[:2]):  # chỉ layer nông
            gram_pred = self.gram_matrix(fp)
            gram_gt   = self.gram_matrix(fg)
            loss_style += self.l1(gram_pred, gram_gt)
        
        # for i in range(len(feat_pred)):
        #     gram_pred = self.gram_matrix(feat_pred[i])
        #     gram_gt = self.gram_matrix(feat_gt[i])
        #     loss_style += self.l1(gram_pred, gram_gt)

        # --- 4. Gradient Loss (Edge) ---
        boundary = (mask > 0.05) & (mask < 0.95)
        loss_grad = self.gradient_loss(refined_img, gt_img)
        loss_grad = torch.mean(loss_grad * boundary)
        # loss_grad = self.gradient_loss(refined_img * mask, gt_img * mask)

        # --- TỔNG HỢP ---
        total_loss = (self.w_l1 * loss_l1) + \
                     (self.w_perc * loss_perc) + \
                     (self.w_style * loss_style) + \
                     (self.w_grad * loss_grad)
                     
        return total_loss, {
            "l1": loss_l1.item(),
            "perc": loss_perc.item(),
            "style": loss_style.item(),
            "grad": loss_grad.item()
        }