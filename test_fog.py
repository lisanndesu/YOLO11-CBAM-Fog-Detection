import torch
import cv2
import numpy as np
import os  # 用于路径处理和文件夹创建

import cv2
import numpy as np
import random

# ========== 雾效参数（可调整） ==========
BETA = 0.5  # 基础雾浓度
A = 230  # 基础大气光值
NOISE_STRENGTH = 0.02  # 雾浓度噪声强度（0.01~0.05）
BLUR_KERNEL = (5, 5)  # 高斯模糊核（模拟雾的散射）
COLOR_SHIFT = (0.02, 0.02, 0.01)  # 雾的轻微偏色（R/G/B）


# ======================================

def haze(img, beta=BETA, A=230):
    # 1. 原图归一化到0~1
    img = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # 2. 提取亮度通道（替代深度图：亮度越高→模拟“越近”→雾越轻）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. 亮度归一化后反转（让亮区域雾轻，暗区域雾浓，贴合俯拍杂草）
    depth = 1 - (gray / (np.max(gray) + 1e-8))  # +1e-8避免除零

    # ========== 增强1：添加雾浓度随机噪声（模拟雾的不均匀性） ==========
    # 生成和图像同尺寸的高斯噪声（均值0，方差NOISE_STRENGTH）
    noise = np.random.normal(0, NOISE_STRENGTH, (h, w))
    # 噪声只作用于雾浓度（避免影响原图），限制范围防止雾效异常
    depth = np.clip(depth + noise, 0.0, 1.0)

    # ========== 增强2：添加局部雾浓度波动（模拟雾团） ==========
    # 生成低频噪声（大核高斯模糊）模拟局部雾团
    fog_patch = np.random.rand(h, w) * 0.1
    fog_patch = cv2.GaussianBlur(fog_patch, (21, 21), 5)  # 大核模糊生成雾团
    depth = np.clip(depth + fog_patch, 0.0, 1.0)

    # 4. 计算透射率（雾浓度），加入随机beta波动（每幅图雾浓度略有不同）
    beta_rand = beta * random.uniform(0.8, 1.2)  # beta±20%随机波动
    trans = np.exp(-beta_rand * depth)

    # 5. 大气光归一化 + 随机波动（模拟雾的亮度不均）
    A_rand = A * random.uniform(0.95, 1.05) / 255.0  # A±5%随机波动
    # 大气光局部微调（模拟雾的亮度渐变）
    A_map = np.ones((h, w)) * A_rand
    A_gradient = np.linspace(0, 0.05, w)  # 轻微亮度渐变
    A_gradient = np.tile(A_gradient, (h, 1))
    A_map = np.clip(A_map + A_gradient, 0.7, 1.0)  # 限制范围

    # ========== 增强3：雾的轻微色彩偏移（真实雾非纯白） ==========
    # 给大气光添加轻微的色彩偏移（贴近田间土黄色调）
    A_rgb = np.ones((h, w, 3)) * A_map[..., None]
    A_rgb[:, :, 0] += COLOR_SHIFT[0]  # 红通道微增
    A_rgb[:, :, 1] += COLOR_SHIFT[1]  # 绿通道微增
    A_rgb[:, :, 2] += COLOR_SHIFT[2]  # 蓝通道微增
    A_rgb = np.clip(A_rgb, 0.0, 1.0)

    # 6. 雾化公式（适配彩色图像）
    hazy = img * trans[..., None] + A_rgb * (1 - trans[..., None])

    # ========== 增强4：高斯模糊（模拟雾的光散射） ==========
    hazy = cv2.GaussianBlur(hazy, BLUR_KERNEL, 1.2)

    # ========== 增强5：添加轻微椒盐噪声（模拟雾中的微小水滴） ==========
    # 椒盐噪声比例控制在0.1%以内，避免失真
    salt_pepper = np.random.choice([0, 1, 2], size=(h, w), p=[0.999, 0.0005, 0.0005])
    hazy[salt_pepper == 1] = 1.0  # 盐噪声（亮斑）
    hazy[salt_pepper == 2] = 0.0  # 椒噪声（暗斑）

    # 7. 还原到0~255，限制范围避免溢出
    hazy = np.clip(hazy * 255, 0, 255).astype(np.uint8)

    return hazy

# 主程序
if __name__ == "__main__":
    # 输入图片路径（原始字符串避免转义）
    input_image_path = r'C:\code_py\yolo11_fog\data\processed\images\val\20210628_iPhoneSE_YL_74.jpg'

    # 保存图片路径（可自定义）
    save_image_path = rf'C:\code_py\yolo11_fog\sereve_hazy_output-{BETA}-test.jpg'

    try:
        # 1. 读取原始图片（关键：先读图片数组，再传给haze函数）
        img = cv2.imread(input_image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{input_image_path}，请检查路径/文件是否存在")

        # 2. 生成雾效
        hazy_image = haze(img, beta=BETA, A=220)  # beta/A可调整雾的浓度/亮度

        # 3. 保存雾效图片（核心保存逻辑）
        # 自动创建保存目录（如果目录不存在）
        save_dir = os.path.dirname(save_image_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存图片到指定路径
        cv2.imwrite(save_image_path, hazy_image)
        print(f"雾效图片已成功保存至：{save_image_path}")

    except Exception as e:
        print(f"运行出错：{e}")
    finally:
        pass