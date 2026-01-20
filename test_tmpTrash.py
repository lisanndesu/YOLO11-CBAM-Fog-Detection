# 激活yolo11-fog环境后运行
import ultralytics
import os

# 获取ultralytics安装目录
ultra_path = os.path.dirname(ultralytics.__file__)
# 拼接配置文件路径
yaml_path = os.path.join(ultra_path, "cfg", "models", "yolo11.yaml")
# 打印路径并验证是否存在
print("YOLO11配置文件路径：", yaml_path)
print("文件是否存在：", os.path.exists(yaml_path))