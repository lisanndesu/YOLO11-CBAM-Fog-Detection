# ============ fog_gate.py ============
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ fog_gate.py ============
# class FogGate(nn.Module):
#     def __init__(self, *_, reduction=16):   # 不再接收 channels
#         super().__init__()
#         # 延迟建立卷积，等第一次 forward 再知道通道数
#         self.reduction = reduction
#         self.t_est = None                    # 占位
#
#     def forward(self, x):
#         if self.t_est is None:               # 第一次建立
#             c = x.size(1)
#             self.t_est = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(c, c // self.reduction, 1, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(c // self.reduction, 1, 1, bias=False)
#             ).to(x.device)
#         # 后续正常计算
#         t = self.t_est(x)
#         gate = torch.sigmoid(1 - t)
#         dark = x.min(dim=1, keepdim=True)[0]
#         out = dark * gate.expand_as(dark) * x
#         return x + out


# ============= ffa_light_gate.py =============
# class FogGate(nn.Module):
#     """
#     轻量 FFA-style 门控：
#     1. 通道分组卷积估计透射率（平滑）
#     2. 残差增强浓雾区域
#     接口仍保持 __init__(*_, reduction=16) 兼容 yaml
#     """
#     def __init__(self, *_, reduction=16, groups=4):
#         super().__init__()
#         self.reduction = reduction
#         self.groups    = groups
#         self.t_est = None   # 延迟建立
#
#     def forward(self, x):
#         if self.t_est is None:
#             c = x.size(1)
#             self.t_est = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 # 分组 1×1 卷积：平滑且参数少
#                 nn.Conv2d(c, c//self.reduction, 1, groups=self.groups, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(c//self.reduction, 1, 1, bias=False)
#             ).to(x.device)
#
#         t   = self.t_est(x)            # [B,1,1,1]
#         gate = torch.sigmoid(1 - t)    # 浓雾→大门控值
#         dark = x.min(dim=1, keepdim=True)[0]
#         out  = dark * gate.expand_as(dark) * x
#         return x + out                 # 残差连接


# 改进方案：「特征级不确定性门控」——不用阈值，让网络自己学
class FogGate(nn.Module):
    def __init__(self, *_, reduction=16):
        super().__init__()
        self.reduction = reduction      # 🔴 存起来
        self.t_est = None
        self.uncertainty = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        if self.t_est is None:
            c = x.size(1)
            self.t_est = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c // self.reduction, 1, bias=False),  # 用 self.reduction
                nn.ReLU(inplace=True),
                nn.Conv2d(c // self.reduction, 1, 1, bias=False)
            ).to(x.device)

        w = torch.sigmoid(self.uncertainty)
        t = self.t_est(x)
        gate = torch.sigmoid(1 - t)
        dark = x.min(dim=1, keepdim=True)[0]
        out = dark * gate.expand_as(dark) * x
        return x + w * out