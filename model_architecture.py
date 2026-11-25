# -*- coding: utf-8 -*-
"""
model_architecture.py
Định nghĩa các kiến trúc mạng phục vụ đồ án Học máy và Ứng dụng:
1. CustomBaselineCNN: Mô hình CNN thuần tự thiết kế để làm mốc đối chuẩn (Baseline Model).
2. PlantMobileNetV2: Mô hình học chuyển giao (Transfer Learning) dựa trên MobileNetV2 - tối ưu cho thiết bị biên và CPU.
"""

import os
import sys

# Đảm bảo đường dẫn cache không bị lỗi font Unicode tiếng Việt (Khương Tử)
os.environ["TORCH_HOME"] = "D:/DoanHMvUD/.cache/torch"

import torch
import torch.nn as nn
import torchvision.models as models

class CustomBaselineCNN(nn.Module):
    """
    Mạng CNN tự thiết kế cơ bản gồm 4 khối tích chập.
    Dùng để so sánh hiệu năng với phương pháp Transfer Learning.
    """
    def __init__(self, num_classes=15):
        super(CustomBaselineCNN, self).__init__()
        self.features = nn.Sequential(
            # Khối 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            # Khối 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            # Khối 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Khối 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 28 -> 14
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_mobilenet_v2_model(num_classes=15, pretrained=True):
    """
    Khởi tạo mô hình MobileNetV2 áp dụng Transfer Learning.
    MobileNetV2 sử dụng Inverted Residual Blocks và Depthwise Separable Convolutions,
    rất nhẹ, suy luận cực nhanh trên CPU mà vẫn đảm bảo độ chính xác vượt trội.
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)
        
    # Đóng băng các lớp trích xuất đặc trưng cơ bản ban đầu
    # để tận dụng tri thức đã học từ ImageNet
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Mở khóa các lớp convolution sâu cuối cùng (layer 14 đến 18) để fine-tuning
    for layer in list(model.features.children())[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Thay thế phần đầu phân loại (Classifier head) cho 15 lớp bệnh cây trồng
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model
