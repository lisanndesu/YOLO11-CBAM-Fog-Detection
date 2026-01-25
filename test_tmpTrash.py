# calc_flops.py - 最终可运行版本
from ultralytics import YOLO
import torch
import warnings

warnings.filterwarnings('ignore')  # 屏蔽无关警告


def calculate_yolo11_gflops(model_path, imgsz=640):
    """
    计算 YOLO11 自定义模型的 GFLOPs（无需推理，直接计算）
    :param model_path: 模型权重文件路径
    :param imgsz: 输入图像尺寸（默认 640）
    :return: (GFLOPs, 参数数量)
    """
    # 加载模型
    model = YOLO(model_path)
    model_fused = model.model.fuse()  # 融合层，保证计算准确
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_fused.to(device)
    model_fused.eval()  # 切换到评估模式

    # 1. 优先用 thop 计算（最准确）
    try:
        from thop import profile, clever_format
        # 创建模拟输入（batch=1, 3通道, 640x640）
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
        # 计算 FLOPs 和参数
        flops, params = profile(model_fused, inputs=(dummy_input,), verbose=False)
        # 转换单位：FLOPs → GFLOPs，params → 百万（M）
        gflops = flops / 1e9
        gflops, params = clever_format([gflops, params], "%.2f")
        return float(gflops.replace('G', '')), params
    except ImportError:
        # 2. 备选方案：无 thop 时用内置方法估算
        print("提示：未安装 thop 库，执行 'pip install thop' 可获得更准确结果")
        flops, params = model_fused.info(verbose=False)[:2]
        gflops = round(flops / 1e9, 2)
        params = f"{params / 1e6:.2f}M"
        return gflops, params


if __name__ == "__main__":
    # 替换为你的模型路径
    MODEL_PATH = r"runs\detect\2cbam_NoCA\weights\best.pt"

    try:
        gflops, params = calculate_yolo11_gflops(MODEL_PATH)
        # 输出结果
        print(f"✅ 模型参数数量: {params}")
        print(f"✅ 模型 GFLOPs (640×640): {gflops}")
    except FileNotFoundError:
        print("❌ 错误：找不到模型文件，请检查路径是否正确！")
    except AttributeError as e:
        print(f"❌ 模型模块错误：{e}")
        print("👉 请先修复 my_cbam.py 中 SpatialAttention 类的 act 属性定义！")
    except Exception as e:
        print(f"❌ 其他错误：{str(e)}")