# ONet_UDE

## 运行

### 安装 uv

```bash
pip install uv
```

### 训练模型

```bash
uv sync
uv run python script/train.py
```

# 训练脚本参数说明

`train.py` (DeepONet) 和 `train_cnn.py` (CNN) 两个训练脚本的所有可选命令行参数。

---

## DeepONet 训练参数

### 基础使用

```bash
python script/train.py [OPTIONS]
```

### 参数列表

| 参数                  | 类型  | 默认值   | 选项                                           | 说明                                                 |
| --------------------- | ----- | -------- | ---------------------------------------------- | ---------------------------------------------------- |
| `--crop`              | flag  | False    | -                                              | 启用裁剪数据集训练                                   |
| `--crop-mode`         | str   | `square` | `boundary`, `random`, `square`, `damage_aware` | 裁剪模式选择                                         |
| `--crop-position`     | str   | `center` | `center`, `corner`, `boundary`, `random`       | square 模式下的裁剪位置                              |
| `--n-keep`            | int   | None     | -                                              | random 模式下保留的传感器数量                        |
| `--damage-threshold`  | float | 0.3      | -                                              | damage_aware 模式下的损伤阈值                        |
| `--min-keep`          | int   | 4        | -                                              | damage_aware 模式下最少保留的传感器数                |
| `--use-subgrid`       | flag  | False    | -                                              | 使用子网格训练模式（10×10→5×5）                      |
| `--img-size`          | int   | 10       | -                                              | 损伤图尺寸（10 或 20）                               |
| `--defect-range-full` | flag  | False    | -                                              | 损伤可出现在整个区域[0,1]（默认[0.2,0.8]）           |
| `--no-crop-input`     | flag  | False    | -                                              | 使用完整传感器网格输入（不裁剪输入，只裁剪监督目标） |

### 使用示例

#### 1. 标准训练（无裁剪）

```bash
python script/train.py
```

- 输入: 5×5×100 传感器信号
- 输出: 10×10 损伤图
- 全量数据训练

#### 2. 正方形裁剪训练

```bash
python script/train.py --crop --crop-mode square --crop-position center
```

- 输入: 从 5×5 裁剪中心 3×3 区域
- 验证 DeepONet 在部分传感器失效时的泛化能力

#### 3. 边界传感器训练

```bash
python script/train.py --crop --crop-mode boundary
```

- 输入: 只使用边界传感器
- 模拟边缘监测场景

#### 4. 随机传感器采样

```bash
python script/train.py --crop --crop-mode random --n-keep 10
```

- 输入: 随机选择 10 个传感器
- 验证稀疏采样性能

#### 5. 基于损伤的自适应裁剪

```bash
python script/train.py --crop --crop-mode damage_aware --damage-threshold 0.3 --min-keep 10
```

- 输入: 移除损伤区域对应的传感器
- 模拟损伤区域传感器失效场景

#### 6. 完整输入+子网格监督 ⭐⭐

```bash
python script/train.py --use-subgrid --img-size 20 --defect-range-full --no-crop-input
```

- 数据生成: 10×10 传感器网格 + 20×20 损伤图
- 训练输入: **完整 10×10 传感器信号**（10000 维）
- 训练监督: 中心 10×10 损伤图
- 测试查询: 可查询完整 20×20 区域
- 对比: 完整信息 vs 外推能力

---

## CNN 训练参数

### 基础使用

```bash
python script/train_cnn.py [OPTIONS]
```

### 参数列表

| 参数                 | 类型  | 默认值   | 选项                                                     | 说明                                  |
| -------------------- | ----- | -------- | -------------------------------------------------------- | ------------------------------------- |
| `--crop`             | flag  | False    | -                                                        | 使用裁剪数据集训练（3×3 网格）        |
| `--crop-position`    | str   | `center` | `center`, `corner`, `boundary`, `random`, `damage_aware` | 裁剪位置选择                          |
| `--damage-threshold` | float | 0.3      | -                                                        | damage_aware 模式下的损伤阈值         |
| `--min-keep`         | int   | 4        | -                                                        | damage_aware 模式下最少保留的传感器数 |
| `--use-subgrid`      | flag  | False    | -                                                        | 使用子网格训练模式（10×10→5×5）       |

### 使用示例

#### 1. 标准 CNN 训练

```bash
python script/train_cnn.py
```

- 输入: 5×5×100 传感器信号
- 输出: 10×10 损伤图
- CNN 卷积架构

#### 2. 中心 3×3 裁剪

```bash
python script/train_cnn.py --crop --crop-position center
```

- 输入: 中心 3×3 区域
- CNN 输入尺寸: 3×3×100

#### 3. 基于损伤的裁剪

```bash
python script/train_cnn.py --crop --crop-position damage_aware --damage-threshold 0.3 --min-keep 4
```

- 输入: 5×5 网格（损伤区域传感器=0）
- 模拟损伤区域传感器失效

#### 4. 子网格训练

```bash
python script/train_cnn.py --use-subgrid
```

- 数据生成: 10×10 传感器网格
- 训练输入: 中心 5×5 区域
- CNN 输入尺寸: 5×5×100

---

## 对比实验

### 实验 1: 查询灵活性对比

**目的**: 展示 DeepONet 的多分辨率查询能力

```bash
# DeepONet: 可查询任意分辨率
python script/train.py --use-subgrid --img-size 20 --defect-range-full

# CNN: 输出尺寸固定
python script/train_cnn.py --use-subgrid
```

**预期结果**:

- DeepONet: 可查询 5×5, 10×10, 20×20 等任意分辨率
- CNN: 只能输出固定尺寸（架构限制）

---

### 实验 2: 空间外推能力

**目的**: 验证 DeepONet 学习连续算子的能力

```bash
# 裁剪输入（信息受限）
python script/train.py --use-subgrid --img-size 20 --defect-range-full

# 完整输入（信息充足）
python script/train.py --use-subgrid --img-size 20 --defect-range-full --no-crop-input
```

**训练监督**: 只监督中心 10×10 区域  
**测试查询**: 查询完整 20×20 区域（包括边缘）

**预期结果**:

- 中心区域: 高精度（训练过）
- 边缘区域: 精度下降但仍能预测（外推能力）
- 完整输入 > 裁剪输入（信息丰富度影响）

---

### 实验 3: 稀疏传感器泛化

**目的**: 对比不同裁剪策略的泛化能力

```bash
# 正方形裁剪
python script/train.py --crop --crop-mode square --crop-position center

# 边界传感器
python script/train.py --crop --crop-mode boundary

# 随机采样
python script/train.py --crop --crop-mode random --n-keep 10

# 损伤自适应
python script/train.py --crop --crop-mode damage_aware --min-keep 10
```

**对比维度**:

- 训练效率
- 测试精度
- 物理合理性

---

## 📊 可视化输出

### DeepONet 训练输出

- `images/train_loss_curve.png` - 训练/测试损失曲线
- `images/train_prediction.png` - 预测结果可视化
- `images/dataset_check/deeponet_cropped_test_sample.png` - 裁剪数据可视化
- `images/subgrid_training_flow.png` - 子网格训练完整流程（15 个子图）⭐
- `images/damage_mapping.png` - 损伤到传感器的映射关系

### CNN 训练输出

- `images/cnn_loss_curve.png` - 训练/测试损失曲线
- `images/cnn_prediction.png` - CNN 预测结果
- `images/dataset_check/cnn_cropped_test.png` - CNN 裁剪数据可视化

---

## 💡 推荐配置

### 快速验证（10 分钟内）

```bash
# DeepONet标准训练
python script/train.py

# CNN标准训练
python script/train_cnn.py
```

### 完整实验（20 分钟内）

```bash
# 1. 查询灵活性实验
python script/train.py --use-subgrid --img-size 20 --defect-range-full
python script/train_cnn.py --use-subgrid

# 2. 外推能力对比
python script/train.py --use-subgrid --img-size 20 --defect-range-full --no-crop-input

# 3. 测试外推性能
python script/test_extrapolation.py
```

---

## 🔧 技术细节

### 数据集配置

- **标准模式**: 5×5 传感器网格, 10×10 损伤图
- **子网格模式**: 10×10 传感器网格, 20×20 损伤图
- **时间步长**: 100 (固定)
- **样本数量**: 2000 (训练集 1600, 测试集 400)

### 网络架构

**DeepONet**:

- Branch 网络: 输入维度取决于传感器数量
  - 标准: 5×5×100 = 2500
  - 子网格(裁剪): 5×5×100 = 2500
  - 子网格(完整): 10×10×100 = 10000
- Trunk 网络: 输入(x, y)坐标, 输出 100 维权重（10×10 的 Imgsize）
- 预测: G(u, y) = Σ bᵢ(u) ψᵢ(y)

**CNN**:

- 输入: (batch, 100, H, W) - 时间作为通道
- 卷积层: 100→64→32→1
- 上采样: 到 10×10 损伤图
- 输出: Sigmoid 激活

### 训练配置

- 优化器: Adam
- 学习率: 5e-4
- 批量大小: 128
- Epochs: 100
- 早停: patience=20

---

## ref

```
@article{lu2021learning,
  title   = {Learning nonlinear operators via {DeepONet} based on the universal approximation theorem of operators},
  author  = {Lu, Lu and Jin, Pengzhan and Pang, Guofei and Zhang, Zhongqiang and Karniadakis, George Em},
  journal = {Nature Machine Intelligence},
  volume  = {3},
  number  = {3},
  pages   = {218--229},
  year    = {2021}
}
```

**更新日期**: 2025-12-17  
**版本**: v2.0
