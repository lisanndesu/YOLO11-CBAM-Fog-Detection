# train_cbam.py
from ultralytics import YOLO
import cbam  # 导入 CBAM 模块，注册到全局

# 加载自定义 YAML，并指定尺度为 n
model = YOLO("models/yolo11_cbam.yaml", cfg={"scale": "n"})

# 开始训练
results = model.train(
    data="data/cottonweed.yaml",
    epochs=50,
    imgsz=640,
    name="clear_cbam",
    device=0
)