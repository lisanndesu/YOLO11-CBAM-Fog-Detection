import cv2, json, random
from pathlib import Path
import numpy as np
import shutil

# ========== 参数 ==========
BETA    = 1.2
OUT_NAME= "fog_val_hard"
# ==========================

ROOT      = Path(__file__).resolve().parents[1]
SRC_IMG   = ROOT / "data/processed/images/val"
SRC_LBL   = ROOT / "data/processed/labels/val"
OUT_IMG   = ROOT / "data" / OUT_NAME
OUT_LBL   = OUT_IMG.with_suffix("")
OUT_IMG.mkdir(exist_ok=True)
OUT_LBL.mkdir(exist_ok=True)

def haze(img, beta=BETA, A=220):
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    dist = np.linalg.norm(np.mgrid[0:h, 0:w] - np.array([[[h//2]], [[w//2]]]), axis=0)
    trans = np.exp(-beta * dist / max(h, w))
    hazy = img * trans[..., None] + A * (1 - trans[..., None])
    return np.clip(hazy, 0, 255).astype(np.uint8)

for img_p in SRC_IMG.glob("*.jpg"):
    # 1. 生成雾图
    img = cv2.imread(str(img_p))
    hzy = haze(img)
    cv2.imwrite(str(OUT_IMG / img_p.name), hzy)

    # 2. 复制同名字标签
    lbl_p = SRC_LBL / f"{img_p.stem}.txt"
    if lbl_p.exists():
        shutil.copy2(lbl_p, OUT_LBL / lbl_p.name)

# 生成 yaml（指向雾图文件夹，标签在同目录 labels/）
yaml_path = ROOT / "data" / f"{OUT_NAME}.yaml"
yaml_path.write_text(f"""
path: C:/code_py/yolo11_fog/data
train: processed/images/train   # 原训练集
val:   fog_val_hard     # 告诉扫描器去子目录找图
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

print(f"✅ hard 雾图 + 标签 已生成 → {OUT_IMG}")