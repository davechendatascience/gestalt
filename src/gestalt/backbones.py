"""From-scratch ResNet backbone (no torchvision dependency) for the 3D-RCNN heads.

The paper uses ResNet-50-C4 (ImageNet-pretrained). On our abstract synthetic
silhouettes ImageNet features don't transfer, so we train a residual backbone
from scratch; the win is depth + residual connections + capacity for the
classification head (the multi-class bottleneck). Widths/blocks are configurable
so it scales from a CPU-friendly ResNet-10 to a GPU ResNet-18/34.
"""
from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.ds = None
        if stride != 1 or cin != cout:
            self.ds = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.ds is None else self.ds(x)
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        return F.relu(o + idt, inplace=True)


class ResNet(nn.Module):
    def __init__(self, in_ch=1, widths=(32, 64, 128, 256), blocks=(1, 1, 1, 1)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                layers.append(BasicBlock(cin, w, stride=2 if (j == 0 and i > 0) else 1))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.out_dim = cin

    def forward(self, x):
        x = self.layers(self.stem(x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)            # (N, out_dim)


def resnet10(in_ch=1):   # CPU-friendly (blocks=1 per stage)
    return ResNet(in_ch, (32, 64, 128, 256), (1, 1, 1, 1))


def resnet18(in_ch=1):
    return ResNet(in_ch, (64, 128, 256, 512), (2, 2, 2, 2))
