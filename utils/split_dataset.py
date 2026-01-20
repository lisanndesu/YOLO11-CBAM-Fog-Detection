import shutil, random
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT   = Path(__file__).resolve().parents[1]
SRC_IMG = ROOT / "data/raw/images"          # 原图
DST_IMG = ROOT / "data/processed/images"    # 目标
LBL_DIR = ROOT / "data/processed/labels"

for split in ("train", "val", "test"):
    (DST_IMG / split).mkdir(parents=True, exist_ok=True)
    (LBL_DIR / split).mkdir(parents=True, exist_ok=True)

# 获取全部 basename
names = [p.stem for p in SRC_IMG.glob("*.jpg")]
random.shuffle(names)

train_n, tmp_n = train_test_split(names, train_size=0.8, random_state=42)
val_n, test_n  = train_test_split(tmp_n, train_size=0.5, random_state=42)

def copy(files, img_src, lbl_src, img_dst, lbl_dst):
    for name in files:
        shutil.copy(img_src / f"{name}.jpg", img_dst / f"{name}.jpg")
        shutil.copy(lbl_src / f"{name}.txt", lbl_dst / f"{name}.txt")

copy(train_n, SRC_IMG, LBL_DIR, DST_IMG / "train", LBL_DIR / "train")
copy(val_n,   SRC_IMG, LBL_DIR, DST_IMG / "val",   LBL_DIR / "val")
copy(test_n,  SRC_IMG, LBL_DIR, DST_IMG / "test",  LBL_DIR / "test")

# 生成 yaml
yaml_path = ROOT / "data/cottonweed.yaml"
yaml_path.write_text(f"""
path: {ROOT / 'data/processed'}
train: images/train
val: images/val
test: images/test
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

print("✅ 划分完成，yaml 已生成 →", yaml_path)