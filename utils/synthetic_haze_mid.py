import cv2, json, random
from pathlib import Path
import numpy as np

IMG_ROOT = Path(r"C:\code_py\yolo11_fog\data\processed\images\val")
OUT_ROOT = Path(r"C:\code_py\yolo11_fog\data\fog_val_med")   # med 档
OUT_ROOT.mkdir(exist_ok=True)

def haze(img, beta=0.8, A=220):
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    dist = np.linalg.norm(np.mgrid[0:h, 0:w] - np.array([[[h//2]], [[w//2]]]), axis=0)
    trans = np.exp(-beta * dist / max(h, w))
    hazy = img * trans[..., None] + A * (1 - trans[..., None])
    return np.clip(hazy, 0, 255).astype(np.uint8)

for img_p in IMG_ROOT.glob("*.jpg"):
    img = cv2.imread(str(img_p))
    hzy = haze(img, beta=0.8)          # med 档
    cv2.imwrite(str(OUT_ROOT / img_p.name), hzy)

# 生成对应 yaml
yaml_txt = f"""
path: C:/code_py/yolo11_fog/data
train: processed/images/train   # 用原训练集
val:   fog_val_med              # 刚才生成的雾图
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
"""
(OUT_ROOT.parent / "fog_med.yaml").write_text(yaml_txt.strip())
print("✅ med 雾图生成完成，yaml →", OUT_ROOT.parent / "fog_med.yaml")