import cv2, json, random
from pathlib import Path
import numpy as np
import shutil

# ========== 参数 ==========
BETA    = 1.7
A = 230
NOISE_STRENGTH = 0.02  # 雾浓度噪声强度（0.01~0.05）
BLUR_KERNEL = (5, 5)  # 高斯模糊核（模拟雾的散射）
COLOR_SHIFT = (0.02, 0.02, 0.01)  # 雾的轻微偏色（R/G/B）
OUT_NAME= f"fog_val_nodeep_{BETA}_{A}"
# ==========================

ROOT      = Path(__file__).resolve().parents[1]
SRC_IMG   = ROOT / "data/processed/images/val"
SRC_LBL   = ROOT / "data/processed/labels/val"
OUT_IMG   = ROOT / "data" / OUT_NAME/"images"
OUT_LBL   = ROOT / "data" / OUT_NAME/ "labels"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

def haze(img, beta:float, A):
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

complite_count = 0
for img_p in SRC_IMG.glob("*.jpg"):
    # 1. 生成雾图
    img = cv2.imread(str(img_p))
    hzy = haze(img, BETA, A)
    cv2.imwrite(str(OUT_IMG / img_p.name), hzy)

    # 2. 复制同名字标签
    lbl_p = SRC_LBL / f"{img_p.stem}.txt"
    if lbl_p.exists():
        shutil.copy2(lbl_p, OUT_LBL / lbl_p.name)
    complite_count += 1
    print(f"comlite count:{complite_count}")

# 生成 yaml（指向雾图文件夹，标签在同目录 labels/）
yaml_path = ROOT / "data" / f"{OUT_NAME}.yaml"
yaml_path.write_text(f"""
path: C:/code_py/yolo11_fog/data
train: processed/images/train   # 原训练集
val:   { OUT_NAME}     # 告诉扫描器去子目录找图
# 标签在同目录 labels/，ultralytics 会自动找
names:
  0: waterhemp
  1: morningglory
  2: purslane
  3: spotted_spurge
  4: carpetweed
  5: ragweed
  6: eclipta
  7: prickly_sida
  8: palm_amaranth
  9: sicklepod
  10: goosegrass
  11: cutleaf_groundcherry
""".strip())

print(f"✅ {OUT_NAME} 雾图 + 标签 已生成 → {OUT_IMG}")