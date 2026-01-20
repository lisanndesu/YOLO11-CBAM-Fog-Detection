# cbam.py - 实现 CBAM 注意力模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力分支
        b, c, h, w = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # 空间注意力分支
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x

# 关键：将 CBAM 模块注册到 YOLO 框架的全局命名空间
from ultralytics.nn.tasks import parse_model
globals()["CBAM"] = CBAM  # 让 parse_model 能找到 CBAM 类