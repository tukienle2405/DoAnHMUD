# -*- coding: utf-8 -*-
"""
explainability.py
Module Explainable AI (XAI) sử dụng kỹ thuật Grad-CAM (Gradient-weighted Class Activation Mapping)
Giúp giải thích mô hình Deep Learning: Trực quan hóa vùng ảnh mà mô hình tập trung quan sát
để đưa ra quyết định phân loại bệnh lá cây.
"""

import os
os.environ["MPLCONFIGDIR"] = "D:/DoanHMvUD/.cache/matplotlib"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.cm as cm

class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Đăng ký hook để lưu activation và gradient
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Sinh bản đồ nhiệt (Heatmap) từ ảnh đầu vào.
        input_tensor: Tensor kích thước (1, C, H, W)
        target_class: Lớp mục tiêu để giải thích (nếu None sẽ lấy lớp có xác suất cao nhất)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Backward pass cho lớp mục tiêu
        score = output[0, target_class]
        score.backward()
        
        # Tính trọng số bằng Global Average Pooling của gradients
        # gradients shape: (1, Channels, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Nhân trọng số với activations và tính tổng có trọng số
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Áp dụng hàm ReLU để chỉ giữ lại các đặc trưng đóng góp tích cực
        cam = F.relu(cam)
        
        # Chuẩn hóa về [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class

    def overlay_heatmap_on_image(self, original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):
        """
        Đè Heatmap dạng màu JET lên ảnh gốc để hiển thị trực quan cho người dùng.
        original_img: PIL Image
        heatmap: numpy 2D array (H_feat, W_feat) giá trị [0, 1]
        alpha: độ trong suốt của heatmap (0.0 đến 1.0)
        """
        w, h = original_img.size
        # Resize heatmap lên cùng kích thước với ảnh gốc
        cam_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        cam_arr = np.array(cam_img) / 255.0
        
        # Áp dụng colormap 'jet'
        try:
            import matplotlib
            colormap = matplotlib.colormaps['jet']
        except Exception:
            colormap = cm.get_cmap('jet')
        color_heatmap = colormap(cam_arr)[:, :, :3]  # Lấy 3 kênh RGB bỏ alpha
        
        orig_arr = np.array(original_img.convert("RGB")) / 255.0
        
        # Trộn màu (Blending)
        blended = (1 - alpha) * orig_arr + alpha * color_heatmap
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended), Image.fromarray((color_heatmap * 255).astype(np.uint8))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
