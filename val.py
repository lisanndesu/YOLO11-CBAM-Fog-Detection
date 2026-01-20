from ultralytics import YOLO

model = YOLO("runs/train/clear/weights/best.pt")
metrics = model.val(data="data/cottonweed.yaml")
print("mAP@50:", metrics.box.map50)