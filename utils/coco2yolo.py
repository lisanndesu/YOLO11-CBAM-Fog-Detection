import json
from pathlib import Path

# ====== 只认你项目路径 ======
ROOT      = Path(__file__).resolve().parents[1]   # -> C:\code_py\yolo11_fog
JSON_PATH = ROOT / "data/raw/weedcoco.json"
IMG_DIR   = ROOT / "data/raw/images"
OUT_DIR   = ROOT / "data/processed"
LBL_DIR   = OUT_DIR / "labels"
# ==========================

LBL_DIR.mkdir(parents=True, exist_ok=True)

coco = json.load(JSON_PATH.open(encoding='utf-8'))
id2file = {i["id"]: i["file_name"] for i in coco["images"]}
id2cat  = {c["id"]: idx for idx, c in enumerate(coco["categories"])}

for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        continue
    img_id   = ann["image_id"]
    cls      = id2cat[ann["category_id"]]
    fname    = id2file[img_id]

    img_w = next(i["width"]  for i in coco["images"] if i["id"] == img_id)
    img_h = next(i["height"] for i in coco["images"] if i["id"] == img_id)
    cx  = (x + w/2) / img_w
    cy  = (y + h/2) / img_h
    nw  = w / img_w
    nh  = h / img_h

    txt_path = LBL_DIR / f"{Path(fname).stem}.txt"
    with txt_path.open("a") as f:
        f.write(f"{cls} {cx} {cy} {nw} {nh}\n")

print(f"✅ 完成！labels → {LBL_DIR}")