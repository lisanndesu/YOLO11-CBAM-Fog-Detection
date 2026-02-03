import torch.nn as nn
import torch



# # ---------------------- 2. 轻量版（愿意掉 0.1-0.2 pp 换速度） ----------------------
# class ChannelAttention(nn.Module):
#     """轻量版：单池化 + 1×1 卷积」"""
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
#         self.act = nn.Sigmoid()
#           print(f"\n\n\n正在使用：自定义CBAM模块（轻量版实现）\n\n\n")
#
#     def forward(self, x):
#         return x * self.act(self.fc(self.avg(x)))
#
#
# class SpatialAttention(nn.Module):
#     """轻量版：3×3 深度可分离"""
#     def __init__(self, kernel_size=3):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         return x * torch.sigmoid(self.conv(x_cat))
#
#
# class CBAM(nn.Module):
#     def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 3):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         return self.sa(self.ca(x))

# ---------------------- 1. 定义CBAM模块（完整实现） ----------------------
# class ChannelAttention(nn.Module):
#     """通道注意力模块（可选择原版双池化或简化版）"""
#
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
#         self.act = nn.Sigmoid()
#         print(f"\n\n\n正在使用：自定义CBAM模块（完整实现）\n\n\n")
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         attn = self.act(avg_out + max_out)
#         return x * attn
#
#
# class SpatialAttention(nn.Module):
#     """空间注意力模块（CBAM必备）"""
#
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         attn = self.act(self.conv(x_cat))
#         return x * attn
#
#
# class CBAM(nn.Module):
#     """完整CBAM模块（通道+空间注意力）"""
#
#     def __init__(self, channels: int, reduction: int = 16, kernel_size=7):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         x = self.ca(x)
#         x = self.sa(x)
#         return x


# ---------------------- 3. NoMaxPool 版（Avg-Only） ----------------------
# class ChannelAttention(nn.Module):
#     """只保留 AvgPool，去掉 MaxPool，参数量减半"""
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
#         self.act = nn.Sigmoid()
#         print(f"\n\n\n正在使用：自定义CBAM模块（Avg-only）\n\n\n")
#     def forward(self, x):
#         att = self.act(self.fc(self.avg(x)))
#         return x * att
#
#
# class SpatialAttention(nn.Module):
#     """原版 7×7 空间注意力"""
#     def __init__(self, kernel_size: int = 7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg = torch.mean(x, dim=1, keepdim=True)
#         max_, _ = torch.max(x, dim=1, keepdim=True)
#         y = torch.cat([avg, max_], dim=1)
#         att = self.act(self.conv(y))
#         return x * att
#
# class CBAM(nn.Module):
#     def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         x = self.ca(x)
#         x = self.sa(x)
#         return x


# ---------------------- 4. NO_SA ----------------------
# class ChannelAttention(nn.Module):
#     """完整双池化通道注意力"""
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.max = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
#         self.act = nn.Sigmoid()
#         print(f"\n\n\n  uisng:CA_ONLY_CBAM   \n\n\n")
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg(x))
#         max_out = self.fc(self.max(x))
#         att = self.act(avg_out + max_out)
#         return x * att
#
#
# class SpatialAttention(nn.Module):
#     """占位 identity，退化为纯通道注意力"""
#     def __init__(self, kernel_size: int = 7):
#         super().__init__()
#
#     def forward(self, x):
#         return x
#
#
# class CBAM(nn.Module):
#     def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         x = self.ca(x)
#         x = self.sa(x)
#         return x


# ---------------------- 5. （NO_CA） ----------------------
# class ChannelAttention(nn.Module):
#     """占位 identity，退化为纯空间注意力"""
#     def __init__(self, channels: int, reduction: int = 16):
#         super().__init__()
#
#     def forward(self, x):
#         return x
#
#
# class SpatialAttention(nn.Module):
#     """原版 7×7 空间注意力"""
#     def __init__(self, kernel_size: int = 7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg = torch.mean(x, dim=1, keepdim=True)
#         max_, _ = torch.max(x, dim=1, keepdim=True)
#         y = torch.cat([avg, max_], dim=1)
#         att = self.act(self.conv(y))
#         return x * att
#
#
# class CBAM(nn.Module):
#     def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(kernel_size)
#         print(f"\n\n\n（SA_only）\n\n\n")
#
#     def forward(self, x):
#         x = self.ca(x)
#         x = self.sa(x)
#         return x


# # ==================================== NoAvg======================================
# class ChannelAttention(nn.Module):
#     def __init__(self, channels: int, reduction:16):
#         super().__init__()
#         self.max = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels//reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels//reduction, channels, 1, bias=False)
#         )
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         return x * self.act(self.fc(self.max(x)))
#
#
# class SpatialAttention(nn.Module):
#     """保持原版 7×7"""
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg = torch.mean(x, dim=1, keepdim=True)
#         max_, _ = torch.max(x, dim=1, keepdim=True)
#         y = torch.cat([avg, max_], dim=1)
#         return x * self.act(self.conv(y))
#
#
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(kernel_size)
#         print(f"\n\n\n  test123  NoAvg   123 \n\n\n")
#
#     def forward(self, x):
#         x = self.ca(x)
#         x = self.sa(x)
#         return x

# -----------------------------------dilated------------------------------------------
# class ChannelAttention(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.max = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels//reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels//reduction, channels, 1, bias=False)
#         )
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg(x))
#         max_out = self.fc(self.max(x))
#         return x * self.act(avg_out + max_out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3, dilations=[1,2,3]):
#         super().__init__()
#         self.dils = nn.ModuleList([
#             nn.Conv2d(2, 1, kernel_size, padding=d, dilation=d, bias=False)
#             for d in dilations
#         ])
#         self.fusion = nn.Conv2d(len(dilations), 1, 1, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         avg = torch.mean(x, dim=1, keepdim=True)
#         max_, _ = torch.max(x, dim=1, keepdim=True)
#         y = torch.cat([avg, max_], dim=1)
#         multi = torch.cat([d(y) for d in self.dils], dim=1)  # [B,3,H,W]
#         att = self.act(self.fusion(multi))
#         return x * att
#
#
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=16, kernel_size=3, dilations=[1,2,3]):
#         super().__init__()
#         self.ca = ChannelAttention(channels, reduction)
#         self.sa = SpatialAttention(kernel_size, dilations)
#         print(f"\n\n\n   using   dilated   CBAM   \n\n\n")
#
#     def forward(self, x):
#         x = self.ca(x)
#         x = self.sa(x)
#         return x

# -----------------------------------2cbam_NoAvg_Dilated------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.max(x))
        return x * self.act(max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, dilations=[1, 2, 3]):
        super().__init__()
        self.dils = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.fusion = nn.Conv2d(len(dilations), 1, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, max_], dim=1)
        multi = torch.cat([d(y) for d in self.dils], dim=1)  # [B,3,H,W]
        att = self.act(self.fusion(multi))
        return x * att

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3, dilations=[1, 2, 3]):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size, dilations)
        print(f"\n\n\n   using   NoAvg_Dilated   CBAM   \n\n\n")

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x