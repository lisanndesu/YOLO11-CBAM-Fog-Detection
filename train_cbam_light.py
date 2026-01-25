from ultralytics import YOLO

# ---------------------- 初始化并训练模型 ----------------------
if __name__ == "__main__":
    # 解决scale警告：YAML中已定义scales，无需传cfg参数
    model = YOLO("models/yolo11_cbam.yaml", verbose=True)
    # 轻量版：channelAttention单池化 + 1×1 卷积   spatialAttention:3×3 深度可分离

    # 训练模型（根据你的数据集配置调整参数）
    results = model.train(          # batch使用默认，同baseline（不指定，由系统自动决定，一般为16，具体真实值可以在runs/clear_cbam/args.yaml中查看（已确认确实为16））
        data="data/cottonweed.yaml",
        epochs=50,  # 同 baseline
        imgsz=640,
        device=0,
        name="clear_cbam_light"  # 输出文件夹，方便对比
    )