import cv2, numpy as np
from pathlib import Path
import shutil, random   # ← 加上 random

ROOT      = Path(__file__).resolve().parents[1]
SRC_IMG   = ROOT / "data/processed/images/val"
OUT_NAME  = "fog_severe"
OUT_IMG   = ROOT / "data" / OUT_NAME
OUT_LBL   = OUT_IMG / "labels"
OUT_IMG.mkdir(exist_ok=True)
OUT_LBL.mkdir(exist_ok=True)

BETA = 2.0
A    = 160

def random_haze(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(3, 5)):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rx = random.randint(w//4, w//2)
        ry = random.randint(h//4, h//2)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=30)
    trans = np.exp(-BETA * mask)
    hazy = img * trans[..., None] + A * (1 - trans[..., None])
    hazy = (hazy * 0.7).astype(np.uint8)
    return np.clip(hazy, 0, 255)

for img_p in SRC_IMG.glob("*.jpg"):
    img = cv2.imread(str(img_p))
    hzy = random_haze(img)
    cv2.imwrite(str(OUT_IMG / img_p.name), hzy)
    lbl_p = SRC_IMG.parent.parent / "labels" / "val" / f"{img_p.stem}.txt"
    if lbl_p.exists():
        shutil.copy2(lbl_p, OUT_LBL / lbl_p.name)

# 生成 yaml（完整内容，直接可用）
yaml_path = ROOT / "data" / "fog_severe.yaml"
yaml_path.write_text(f"""path: {ROOT / 'data'}
train: processed/images/train
val:   fog_severe
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

print("✅ severe 雾图 + 标签 已生成 →", OUT_IMG)