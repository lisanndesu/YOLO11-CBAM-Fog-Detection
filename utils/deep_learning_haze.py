import torch
import cv2
import numpy as np
import os  # 新增：用于路径处理和文件夹创建

# 加载MiDaS深度估计模型
model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)  # 补充trust_repo消除警告
model.eval()

# 强制使用CPU运行（避免显存不足）
device = torch.device("cpu")
model.to(device)


def add_haze(image_path, save_path=None):
    """
   给图片添加雾效，并可选保存结果
   :param image_path: 输入图片的路径
   :param save_path: 雾效图片的保存路径（None则不保存）
   :return: 加雾后的图片数组
   """
    # 1. 读取图片并校验
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}，请检查路径或文件是否存在")

    # 2. 模型输入预处理（关键：补充MiDaS要求的标准化，否则会报错）
    # 原代码缺少预处理，这里补充（否则模型推理会因输入格式错误失败）
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),  # DPT_Hybrid要求的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转RGB（MiDaS要求）
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # 3. 生成深度图
    with torch.no_grad():
        depth_map = model(input_tensor)

    # 4. 处理深度图（匹配原图尺寸）
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))  # 匹配原图尺寸
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    # 5. 生成雾效
    fog_intensity = 1 - depth_map
    fog_layer = np.ones_like(image) * 200
    hazy_image = image * (1 - fog_intensity[..., np.newaxis]) + fog_layer * fog_intensity[..., np.newaxis]
    hazy_image = np.clip(hazy_image, 0, 255).astype(np.uint8)

    # 6. 保存雾效图片（新增核心功能）
    if save_path is not None:
        # 自动创建保存目录（如果不存在）
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存图片
        cv2.imwrite(save_path, hazy_image)
        print(f"雾效图片已保存至：{save_path}")

    return hazy_image


# 主程序
if __name__ == "__main__":
    # 输入图片路径（原始字符串避免转义）
    input_image_path = r'C:\code_py\yolo11_fog\data\processed\images\val\20210628_iPhoneSE_YL_74.jpg'

    # 保存图片路径（可自定义，比如保存到代码同目录）
    save_image_path = r'C:\code_py\yolo11_fog\sereve_hazy_output.jpg'

    try:
        # 生成雾效并保存
        hazy_image = add_haze(input_image_path, save_path=save_image_path)

        # 显示雾效图片
        cv2.imshow('Hazy Image', hazy_image)
        cv2.waitKey(0)  # 按任意键关闭窗口
    except Exception as e:
        print(f"运行出错：{e}")
    finally:
        cv2.destroyAllWindows()  # 确保窗口关闭