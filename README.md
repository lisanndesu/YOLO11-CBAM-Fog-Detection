# YOLO11-CBAM-Fog-Detection
A robust weed detection model optimized for low-visibility scenarios (morning fog, dust fog, cloudy days) based on YOLO11 and CBAM attention mechanism.

## 项目简介
本项目针对棉花田杂草检测在复杂天气下精度下降的问题，基于YOLO11模型引入自定义CBAM（Convolutional Block Attention Module）注意力机制，优化模型在晨雾、尘雾、阴天等低能见度场景下的检测性能。通过合成不同浓度的雾化数据集进行专项测试，验证了改进方案在不损失清晰场景性能的前提下，显著提升了雾天环境下的杂草检测鲁棒性，为农业无人机精准喷药等实际应用提供技术支持。

## 核心特性
- **技术架构**：基于YOLO11框架，集成自定义CBAM注意力模块（非内置版本，优化特征聚焦能力）
- **场景适配**：专门优化晨雾、尘雾、阴天等低能见度环境检测性能
- **数据支持**：自动合成多浓度雾化数据集，支持8:1:1自动划分训练/验证/测试集
- **性能优势**：雾天场景下mAP50最高提升4.1个百分点，兼顾检测精度与实时性
- **工程化适配**：提供完整的数据集处理、模型训练、性能验证流程，支持快速复现

## 环境配置
```bash
# 克隆仓库
git clone https://github.com/你的用户名/yolo11-cbam-fog-detection.git
cd yolo11-cbam-fog-detection

# 安装依赖（建议使用conda环境）
conda create -n yolo11-fog python=3.9
conda activate yolo11-fog
pip install ultralytics==8.3.11 torch torchvision numpy==2.2.6
```

## 数据集说明
### 原始数据集
- 基础数据集：CottonWeed杂草检测数据集（12类目标，`nc=12`）
- 数据格式：YOLO格式（图像+标注文件），自动划分比例8:1:1

### 雾化数据集生成
通过大气散射模型合成不同浓度的雾化数据集，支持以下类型：
| 雾化类型 | 核心参数 | 对应脚本 |
|----------|----------|----------|
| 中度雾 | beta=0.8 | `utils/synthetic_haze_mid.py` |
| 重度雾 | beta=1.2 | `utils/synthetic_haze_hard.py` |
| 极端雾 | - | `utils/synthetic_sereve.py` |
| 无深度雾 | beta=0.5/1.2/1.7, A=230 | `utils/synthetic_haze_nodeep_*.py` |

### 数据集处理流程
```bash
# 1. COCO格式转YOLO格式
python utils/coco2yolo.py

# 2. 自动划分数据集+生成配置文件（cottonweed.yaml）
python utils/split_dataset.py

# 3. 合成雾化数据集（以无深度雾为例）
python utils/synthetic_haze_nodeep_1.7_230.py

# 4. 标注文件复制（按需执行）
xcopy /s /y C:\code_py\yolo11_fog\data\processed\labels\val C:\code_py\yolo11_fog\data\fog_val_nodeep_1.7_230\
```

## 模型训练与验证
### 1. 基准模型训练（无CBAM）
```bash
yolo train data=data/cottonweed.yaml model=yolo11n.yaml epochs=50 name=clear
```
- 清晰场景性能：mAP50=0.84292，mAP50-95=0.76415

### 2. CBAM改进模型训练
#### 模型配置
- 下载YOLO11基础配置：[yolo11.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml)
- 新增CBAM模块：在Neck部分P3层插入 `- [-1, 1, CBAM, []]`
- 保存为：`models/yolo11_cbam.yaml`（修改`nc=12`）

#### CBAM模块集成
需手动配置YOLO11框架支持自定义CBAM（详细步骤见[配置指南](https://www.doubao.com/thread/w26a96e93ade43011)）：
1. 在`ultralytics/nn/modules/`下新建`my_cbam.py`（自定义CBAM实现）
2. 修改`modules/__init__.py`，导入自定义CBAM并屏蔽内置版本
3. 在`tasks.py`的`attempt_load_one_weight`函数中注册CBAM类

#### 启动训练
```bash
python train_cbam.py
```
- 清晰场景性能：mAP50=0.849，mAP50-95=0.769（略优于基准模型）

### 3. 雾天场景验证
```bash
# 以无深度雾（beta=1.7, A=230）为例
yolo val data=data/fog_val_nodeep_1.7_230.yaml model=runs/detect/clear_cbam/weights/best.pt
```

## 实验结果
### 不同雾浓度下性能对比（mAP）
| 测试场景 | 模型 | mAP50 | mAP50-95 | 性能提升（mAP50） |
|----------|------|-------|----------|-------------------|
| 清晰场景 | YOLO11（基准） | 0.84292 | 0.76415 | - |
| 清晰场景 | YOLO11-CBAM | 0.849 | 0.769 | +0.6个百分点 |
| 无深度雾（beta=0.5） | YOLO11（基准） | 0.825 | 0.754 | - |
| 无深度雾（beta=0.5） | YOLO11-CBAM | 0.829 | 0.754 | +0.4个百分点 |
| 无深度雾（beta=1.2） | YOLO11（基准） | 0.793 | 0.726 | - |
| 无深度雾（beta=1.2） | YOLO11-CBAM | 0.82 | 0.747 | +2.7个百分点 |
| 无深度雾（beta=1.7） | YOLO11（基准） | 0.771 | 0.705 | - |
| 无深度雾（beta=1.7） | YOLO11-CBAM | 0.812 | 0.74 | +4.1个百分点 |

### 关键结论
1. 随着雾浓度增加，基准模型性能持续下降，而YOLO11-CBAM的性能衰减速率显著更低
2. 在高浓度雾场景（beta=1.7）下，改进模型的mAP50提升最为明显（+4.1个百分点）
3. 清晰场景下，CBAM模块未造成性能损失，反而略有提升，验证了改进的有效性

## 项目结构
```
yolo11-cbam-fog-detection/
├── data/                  # 数据集目录（不含原始数据，通过脚本生成）
│   ├── cottonweed.yaml    # 数据集配置文件
│   ├── fog_val_*/         # 各类雾化验证集
│   └── processed/         # 处理后的数据集（自动生成）
├── models/                # 模型配置文件
│   ├── yolo11.yaml        # YOLO11基准配置
│   └── yolo11_cbam.yaml   # CBAM改进模型配置
├── runs/                  # 训练日志与权重（.gitignore忽略）
├── utils/                 # 工具脚本
│   ├── coco2yolo.py       # 格式转换脚本
│   ├── split_dataset.py   # 数据集划分脚本
│   └── synthetic_*.py     # 雾化数据集生成脚本
├── train_cbam.py          # CBAM模型训练脚本
├── .gitignore             # Git忽略文件配置
└── README.md              # 项目说明文档
```

## 注意事项
1. 数据集目录`data/`已加入`.gitignore`，需通过`utils/`目录下脚本自行生成
2. CBAM模块为自定义实现，需严格按照配置指南修改YOLO11源码，否则会出现加载错误
3. 雾化数据集生成时，确保原始标注文件路径正确，避免标注丢失
4. 训练时建议使用GPU（device=0），否则训练速度会显著变慢

## 未来优化方向
1. 集成轻量级去雾门控模块，进一步提升雾天场景特征提取能力
2. 引入知识蒸馏技术，实现模型轻量化，适配边缘设备部署
3. 补充红外图像数据，实现RGB+IR双光谱融合检测，覆盖夜间低光场景
4. 增加真实雾天数据集验证，提升模型实际应用可靠性

## 许可证
本项目基于AGPL-3.0许可证开源（遵循Ultralytics YOLO许可证协议），仅供学术研究与非商业用途。使用本项目代码请注明出处。
