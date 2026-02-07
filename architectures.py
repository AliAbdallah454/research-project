import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import numpy as np
import cv2
from PIL import Image

from typing import Tuple, Callable, Optional

class CircleRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out