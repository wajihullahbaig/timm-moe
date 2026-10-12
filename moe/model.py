"""Backbone -> gate + experts -> classifier."""
from typing import NamedTuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEOutput(NamedTuple):
    logits: torch.Tensor          # (B, num_classes)
    gate_weights: torch.Tensor    # (B, num_experts), rows sum to 1
    expert_outputs: torch.Tensor  # (B, num_experts, C, H, W)


class ExpertBank(nn.Module):
    """`num_experts` residual conv experts computed in one fused pass.

    The experts share their input, so the first conv is a single wide conv and the second is
    grouped; this is mathematically E independent experts but runs as two kernels, not 2E.
    """

    def __init__(self, dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts, self.dim = num_experts, dim
        wide = num_experts * hidden_dim
        self.conv1 = nn.Conv2d(dim, wide, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(wide)
        self.conv2 = nn.Conv2d(wide, num_experts * dim, 3, padding=1, groups=num_experts)
        self.bn2 = nn.BatchNorm2d(num_experts * dim)
        self.skip = nn.Sequential(nn.Conv2d(dim, num_experts * dim, 1), nn.BatchNorm2d(num_experts * dim))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x)).unflatten(1, (self.num_experts, self.dim))


class Gate(nn.Module):
    """Spatial attention, avg+max pooling, then an MLP producing softmax(logits / temperature)."""

    def __init__(self, dim, num_experts, temperature, dropout=0.3):
        super().__init__()
        self.temperature = temperature
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.BatchNorm2d(dim // 4), nn.ReLU(),
            nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim // 2, num_experts),
        )

    def forward(self, x):
        x = x * self.spatial(x)
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        return F.softmax(self.mlp(pooled).float() / self.temperature, dim=1)


class MoEClassifier(nn.Module):
    def __init__(self, num_experts, num_classes=10, expert_hidden_dim=128, temperature=1.0,
                 backbone='resnet18', out_index=2, input_scale=2, pretrained=True):
        super().__init__()
        self.input_scale = input_scale
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(out_index,))
        dim = self.backbone.feature_info.channels()[-1]
        self.gate = Gate(dim, num_experts, temperature)
        self.experts = ExpertBank(dim, expert_hidden_dim, num_experts)
        self.classifier = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        if self.input_scale != 1:  # 32px inputs starve pretrained backbones; upsampling is the biggest single gain
            x = F.interpolate(x, scale_factor=self.input_scale, mode='bilinear', align_corners=False)
        features = self.backbone(x)[0]
        gate_weights = self.gate(features)
        expert_outputs = self.experts(features)
        combined = torch.einsum('be,bechw->bchw', gate_weights.to(expert_outputs.dtype), expert_outputs)
        return MoEOutput(self.classifier(combined), gate_weights, expert_outputs)
