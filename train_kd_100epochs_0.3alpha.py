from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    # 1. 教师模型 (确保 device 统一)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = YOLO(r"runs/detect/2cbam_autoFogGate/weights/best.pt")
    teacher.model.to(device)
    teacher.model.eval()
    for p in teacher.model.parameters():
        p.requires_grad = False

    # 2. 学生模型
    student = YOLO(r"models/miniYolo11n.yaml")

    # 3. 定义蒸馏损失计算函数
    def dist_loss_callback(trainer):
        # A. 正确获取数据 (trainer.batch 是字典)
        imgs = trainer.batch['img'].float() / 255.0

        # B. 教师前向传播 (关闭梯度)
        with torch.no_grad():
            # 获取教师的中间特征层或最后输出
            # teacher.model(imgs) 在 eval 模式下返回 (inference_out, loss_out)
            t_out = teacher.model(imgs)[0]

            # C. 学生前向传播
        # 注意：trainer.model 是原始的 nn.Module
        s_out = trainer.model(imgs)[0]

        # D. 简单的 KL 散度实现 (针对分类 Logits)
        # 注意：由于 YOLO 输出包含 Box 和 Cls，直接全局 KL 并不严谨
        # 这里仅演示如何修复报错并挂载损失
        T = 3.0
        # 确保形状对齐（如果 mini 版本宽度不同，需要插值或投影）
        if s_out.shape != t_out.shape:
            s_out_resized = F.interpolate(s_out, size=t_out.shape[2:], mode='bilinear')
        else:
            s_out_resized = s_out

        kl_loss = F.kl_div(
            F.log_softmax(s_out_resized / T, dim=1),
            F.softmax(t_out / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)

        # E. 将蒸馏损失叠加到总损失上 (核心：必须在 backward 之前)
        alpha = 0.3
        trainer.loss *= alpha
        trainer.loss += kl_loss * (1 - alpha)

    # 4. 挂载回调
    # 使用 on_after_compute_loss 才能让损失参与反向传播
    student.add_callback("on_after_compute_loss", dist_loss_callback)

    # 5. 开始训练
    student.train(
        data=r"data/cottonweed.yaml",
        epochs=100,
        name="KD_miniYolo11n_100epochs_0.3alpha",
        device=0,
        workers=4,  # Windows下设为0防止多线程错误
        imgsz=640
    )


if __name__ == '__main__':
    main()
