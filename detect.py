from ultralytics import YOLO

model = YOLO("runs/train/clear/weights/best.pt")
model.predict(source="data/processed/images/val",
              save=True,
              name="val_clear")