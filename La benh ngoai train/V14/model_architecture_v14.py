# -*- coding: utf-8 -*-
"""Architecture identical to the MobileNetV2 checkpoints used by V12C."""
import torch
from torch import nn
from torchvision import models


def get_mobilenet_v2_model(num_classes, pretrained=False):
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return m
