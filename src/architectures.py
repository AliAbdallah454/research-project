import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from typing import Tuple, Callable, Optional
    
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class CircleRegressorResNet(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, out_dim: int = 6):
        super().__init__()

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out
