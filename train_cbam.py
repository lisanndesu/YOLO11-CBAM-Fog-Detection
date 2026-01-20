from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import torch.nn as nn
import torch

#   后期已在nn/model包中添加并配置了my_cbam.py， 所以这部分配置CBAM的代码可以删去了
# # ---------------------- 1. 定义CBAM模块（完整实现） ----------------------
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
#
#
# # ---------------------- 2. 注册CBAM到Ultralytics框架 ----------------------
# # 将CBAM类添加到tasks模块的全局变量中，让解析器能找到
# tasks.CBAM = CBAM
# # 额外兜底：注册到parse_model函数的globals()（可选，确保万无一失）
# globals()["CBAM"] = CBAM

# ---------------------- 3. 初始化并训练模型 ----------------------
if __name__ == "__main__":
    # 解决scale警告：YAML中已定义scales，无需传cfg参数
    model = YOLO("models/yolo11_cbam.yaml", verbose=True)

    # 训练模型（根据你的数据集配置调整参数）
    results = model.train(          # batch使用默认，同baseline（不指定，由系统自动决定，一般为16，具体真实值可以在runs/clear_cbam/args.yaml中查看（已确认确实为16））
        data="data/cottonweed.yaml",
        epochs=50,  # 同 baseline
        imgsz=640,
        device=0,
        name="clear_cbam"  # 输出文件夹，方便对比
    )