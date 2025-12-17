"""
数据变换模块 - 用于验证DeepONet的泛化能力

功能：
1. 空间裁剪：保留边界传感器，移除中心区域
2. 随机采样：每个样本随机选择传感器位置
3. 自动适配Branch网络输入维度
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
import random
import matplotlib.pyplot as plt
import os


class SpatialCropper:
    """
    空间裁剪器 - 从完整网格中选择部分传感器
    
    用途：验证DeepONet能否从不完整观测预测损伤
    """
    
    def __init__(
        self,
        nx: int = 5,
        ny: int = 5,
        sig_len: int = 100,
        crop_mode: str = 'boundary',  # 'boundary' 或 'random'
        n_keep: Optional[int] = None,  # 保留的传感器数量（random模式）
        random_seed: Optional[int] = None
    ):
        """
        Args:
            nx, ny: 原始空间网格大小
            sig_len: 时间步长
            crop_mode: 'boundary' - 保留边界, 'random' - 随机采样
            n_keep: random模式下保留的传感器数量
            random_seed: 随机种子（用于复现）
        """
        self.nx = nx
        self.ny = ny
        self.sig_len = sig_len
        self.crop_mode = crop_mode
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 计算边界点索引
        self.boundary_indices = self._get_boundary_indices()
        
        # 确定保留点数量
        if crop_mode == 'boundary':
            self.n_keep = len(self.boundary_indices)
        else:  # random
            self.n_keep = n_keep if n_keep is not None else len(self.boundary_indices)
        
        print(f"✓ SpatialCropper initialized:")
        print(f"  - Mode: {crop_mode}")
        print(f"  - Original grid: {ny}×{nx} = {nx*ny} sensors")
        print(f"  - Kept sensors: {self.n_keep}")
        print(f"  - Input dim: {nx*ny*sig_len} → {self.n_keep*sig_len}")
    
    def _get_boundary_indices(self) -> List[Tuple[int, int]]:
        """
        获取边界点的(y, x)索引
        
        边界定义：第一行、最后一行、第一列、最后一列
        """
        indices = []
        
        # 第一行和最后一行
        for x in range(self.nx):
            indices.append((0, x))  # 第一行
            if self.ny > 1:
                indices.append((self.ny - 1, x))  # 最后一行
        
        # 第一列和最后一列（排除角点，已在上面添加）
        for y in range(1, self.ny - 1):
            indices.append((y, 0))  # 第一列
            if self.nx > 1:
                indices.append((y, self.nx - 1))  # 最后一列
        
        # 去重（针对小网格）
        indices = list(set(indices))
        
        return sorted(indices)
    
    def crop_signal(
        self,
        signal: np.ndarray,
        random_per_sample: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        裁剪信号
        
        Args:
            signal: (ny, nx, sig_len) 或 (batch, ny, nx, sig_len)
            random_per_sample: random模式下是否每个样本独立随机
        
        Returns:
            cropped_signal: (n_keep, sig_len) 或 (batch, n_keep, sig_len)
            kept_indices: 保留的传感器索引列表
        """
        is_batch = signal.ndim == 4
        
        if not is_batch:
            signal = signal[np.newaxis, ...]  # 增加batch维度
        
        batch_size = signal.shape[0]
        cropped_signals = []
        all_indices = []
        
        for i in range(batch_size):
            sig_single = signal[i]  # (ny, nx, sig_len)
            
            # 选择传感器索引
            if self.crop_mode == 'boundary':
                kept_indices = self.boundary_indices
            else:  # random
                if random_per_sample or i == 0:
                    # 从所有点中随机选择
                    all_points = [(y, x) for y in range(self.ny) for x in range(self.nx)]
                    kept_indices = random.sample(all_points, self.n_keep)
                # 否则使用第一个样本的索引
            
            # 提取信号
            cropped_sig = np.array([sig_single[y, x, :] for y, x in kept_indices])
            cropped_signals.append(cropped_sig)
            all_indices.append(kept_indices)
        
        cropped_signals = np.stack(cropped_signals, axis=0)
        
        if not is_batch:
            cropped_signals = cropped_signals[0]
            all_indices = all_indices[0]
        
        return cropped_signals, all_indices
    
    def get_branch_dim(self) -> int:
        """返回裁剪后的branch网络输入维度"""
        return self.n_keep * self.sig_len
    
    def visualize_crop_pattern(self) -> np.ndarray:
        """
        可视化裁剪模式
        
        Returns:
            mask: (ny, nx) 二值矩阵，1表示保留的点
        """
        mask = np.zeros((self.ny, self.nx), dtype=int)
        
        if self.crop_mode == 'boundary':
            indices = self.boundary_indices
        else:
            # random模式显示一个示例
            all_points = [(y, x) for y in range(self.ny) for x in range(self.nx)]
            indices = random.sample(all_points, self.n_keep)
        
        for y, x in indices:
            mask[y, x] = 1
        
        return mask


class CroppedDatasetWrapper:
    """
    裁剪数据集包装器
    
    将原始数据集包装，返回裁剪后的信号
    """
    
    def __init__(
        self,
        base_dataset,
        cropper: SpatialCropper,
        random_per_sample: bool = True
    ):
        """
        Args:
            base_dataset: 原始数据集（如SimpleUSDataset3D）
            cropper: 裁剪器
            random_per_sample: 是否每个样本独立随机裁剪
        """
        self.base_dataset = base_dataset
        self.cropper = cropper
        self.random_per_sample = random_per_sample
        
        print(f"✓ CroppedDatasetWrapper created:")
        print(f"  - Base dataset: {len(base_dataset)} samples")
        print(f"  - Random per sample: {random_per_sample}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            cropped_signal: (n_keep, sig_len) - 裁剪后的信号
            image: (img_size, img_size) - 完整的损伤图（不变）
        """
        signal, image = self.base_dataset[idx]
        
        # 裁剪信号
        cropped_signal, _ = self.cropper.crop_signal(
            signal,
            random_per_sample=self.random_per_sample
        )
        
        return cropped_signal, image
    
    def get_branch_dim(self) -> int:
        """返回branch网络输入维度"""
        return self.cropper.get_branch_dim()


class SquareCropper:
    """
    正方形裁剪器 - 保持CNN友好的正方形输入格式
    
    用途：从 5×5 网格裁剪到 3×3 网格，保留空间结构
    """
    
    def __init__(
        self,
        nx: int = 5,
        ny: int = 5,
        sig_len: int = 100,
        crop_size: int = 3,  # 裁剪后的正方形尺寸
        crop_position: str = 'center',  # 'center', 'corner', 'random'
        random_seed: Optional[int] = None
    ):
        """
        Args:
            nx, ny: 原始网格尺寸
            sig_len: 时间步长
            crop_size: 裁剪后尺寸（如 3×3）
            crop_position: 裁剪位置
                - 'center': 保留中心区域
                - 'corner': 保留左上角
                - 'boundary': 保留边界和中心（3×3时为边界+中心点）
                - 'random': 随机位置
            random_seed: 随机种子
        """
        assert crop_size < min(nx, ny), "Crop size must be smaller than grid"
        assert nx == ny, "SquareCropper requires square input grid"
        
        self.nx = nx
        self.ny = ny
        self.sig_len = sig_len
        self.crop_size = crop_size
        self.crop_position = crop_position
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 计算裁剪索引
        self.crop_indices = self._get_crop_indices()
        
        print(f"✓ SquareCropper initialized:")
        print(f"  - Original grid: {ny}×{nx}")
        print(f"  - Cropped grid: {crop_size}×{crop_size}")
        print(f"  - Crop position: {crop_position}")
        print(f"  - Retention rate: {crop_size**2}/{nx*ny} = {crop_size**2/(nx*ny)*100:.1f}%")
        print(f"  - Signal shape: ({ny}, {nx}, {sig_len}) → ({crop_size}, {crop_size}, {sig_len})")
    
    def _get_crop_indices(self) -> List[Tuple[int, int]]:
        """
        获取裁剪区域的索引
        
        Returns:
            indices: [(y0, x0), (y0, x1), ...] 保留点的索引
        """
        indices = []
        
        if self.crop_position == 'center':
            # 中心裁剪
            start = (self.nx - self.crop_size) // 2
            for y in range(start, start + self.crop_size):
                for x in range(start, start + self.crop_size):
                    indices.append((y, x))
        
        elif self.crop_position == 'corner':
            # 左上角
            for y in range(self.crop_size):
                for x in range(self.crop_size):
                    indices.append((y, x))
        
        elif self.crop_position == 'boundary':
            # 边界 + 中心点（适用于3×3）
            # 策略：保留四角 + 四边中点 + 中心 = 9个点
            if self.crop_size == 3 and self.nx == 5:
                # 规则排列为3×3网格
                indices = [
                    (0, 0), (0, 2), (0, 4),    # 上边界
                    (2, 0), (2, 2), (2, 4),    # 中间行
                    (4, 0), (4, 2), (4, 4)     # 下边界
                ]
            else:
                # 通用边界策略
                step = (self.nx - 1) // (self.crop_size - 1)
                for i in range(self.crop_size):
                    for j in range(self.crop_size):
                        indices.append((i * step, j * step))
        
        elif self.crop_position == 'random':
            # 随机选择crop_size×crop_size个不重复点
            all_points = [(y, x) for y in range(self.ny) for x in range(self.nx)]
            selected = random.sample(all_points, self.crop_size ** 2)
            # 排序以保持某种空间结构
            indices = sorted(selected)
        
        else:
            raise ValueError(f"Unknown crop_position: {self.crop_position}")
        
        return indices
    
    def crop_signal(
        self,
        signal: np.ndarray,
        return_grid: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        裁剪信号，保持正方形格式
        
        Args:
            signal: (ny, nx, sig_len) 或 (batch, ny, nx, sig_len)
            return_grid: 是否返回网格格式（True）还是flatten（False）
        
        Returns:
            cropped_signal: 
                - 如果return_grid=True: (crop_size, crop_size, sig_len) 或 (batch, crop_size, crop_size, sig_len)
                - 如果return_grid=False: (crop_size^2, sig_len) 或 (batch, crop_size^2, sig_len)
            kept_indices: 保留的索引列表
        """
        is_batch = signal.ndim == 4
        
        if not is_batch:
            signal = signal[np.newaxis, ...]
        
        batch_size = signal.shape[0]
        
        if return_grid:
            # 返回网格格式
            cropped_signals = np.zeros(
                (batch_size, self.crop_size, self.crop_size, self.sig_len),
                dtype=signal.dtype
            )
            
            for b in range(batch_size):
                for i, (y, x) in enumerate(self.crop_indices):
                    cy = i // self.crop_size
                    cx = i % self.crop_size
                    cropped_signals[b, cy, cx, :] = signal[b, y, x, :]
        else:
            # 返回flatten格式
            cropped_signals = []
            for b in range(batch_size):
                cropped_sig = np.array([signal[b, y, x, :] for y, x in self.crop_indices])
                cropped_signals.append(cropped_sig)
            cropped_signals = np.stack(cropped_signals, axis=0)
        
        if not is_batch:
            cropped_signals = cropped_signals[0]
        
        return cropped_signals, self.crop_indices
    
    def visualize_crop_pattern(self) -> np.ndarray:
        """
        可视化裁剪模式
        
        Returns:
            mask: (ny, nx) 二值矩阵
        """
        mask = np.zeros((self.ny, self.nx), dtype=int)
        for y, x in self.crop_indices:
            mask[y, x] = 1
        return mask


class SquareCroppedDatasetWrapper:
    """
    正方形裁剪数据集包装器 - 适用于CNN
    """
    
    def __init__(
        self,
        base_dataset,
        cropper: SquareCropper,
        for_cnn: bool = True  # True返回网格，False返回flatten
    ):
        """
        Args:
            base_dataset: 原始数据集
            cropper: 正方形裁剪器
            for_cnn: 是否为CNN准备数据（保持网格格式）
        """
        self.base_dataset = base_dataset
        self.cropper = cropper
        self.for_cnn = for_cnn
        
        print(f"✓ SquareCroppedDatasetWrapper created:")
        print(f"  - Base dataset: {len(base_dataset)} samples")
        print(f"  - For CNN: {for_cnn}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        signal, image = self.base_dataset[idx]
        
        # 裁剪信号
        cropped_signal, _ = self.cropper.crop_signal(
            signal,
            return_grid=self.for_cnn
        )
        
        return cropped_signal, image
    
    def get_crop_size(self):
        """返回裁剪后的网格尺寸"""
        return self.cropper.crop_size
    
    def get_branch_dim(self):
        """返回branch维度（用于DeepONet）"""
        return self.cropper.crop_size ** 2 * self.cropper.sig_len


class DamageAwareCropper:
    """
    基于损伤位置的自适应裁剪器
    
    原理：
    1. 根据损伤概率图识别损伤区域（阈值 > 0.3）
    2. 将损伤区域映射到最近的传感器网格
    3. 移除损伤区域对应的传感器
    4. 保留其余传感器，至少保留4个
    
    用途：模拟"移除损伤区域传感器"的真实场景
    """
    
    def __init__(
        self,
        nx: int = 5,
        ny: int = 5,
        sig_len: int = 100,
        img_size: int = 10,
        damage_threshold: float = 0.3,
        min_keep: int = 4,
        random_seed: Optional[int] = None
    ):
        """
        Args:
            nx, ny: 传感器网格尺寸
            sig_len: 时间步长
            img_size: 损伤图尺寸
            damage_threshold: 损伤阈值（概率）
            min_keep: 最少保留的传感器数量
            random_seed: 随机种子
        """
        self.nx = nx
        self.ny = ny
        self.sig_len = sig_len
        self.img_size = img_size
        self.damage_threshold = damage_threshold
        self.min_keep = min_keep
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        print(f"✓ DamageAwareCropper initialized:")
        print(f"  - Sensor grid: {ny}×{nx}")
        print(f"  - Damage image: {img_size}×{img_size}")
        print(f"  - Damage threshold: {damage_threshold}")
        print(f"  - Min sensors kept: {min_keep}")
    
    def _map_damage_to_sensors(self, damage_image: np.ndarray) -> List[Tuple[int, int]]:
        """
        将损伤区域映射到传感器网格（最近邻）
        
        Args:
            damage_image: (img_size, img_size) 损伤概率图
        
        Returns:
            damaged_sensors: [(y, x), ...] 损伤区域对应的传感器索引
        """
        # 1. 找出损伤区域（概率 > 阈值）
        damage_mask = damage_image > self.damage_threshold  # (img_size, img_size)
        
        # 2. 构建映射：损伤图坐标 → 传感器坐标
        # 损伤图网格点
        img_y, img_x = np.where(damage_mask)
        
        if len(img_y) == 0:
            # 无损伤区域
            return []
        
        # 归一化到 [0, 1]
        img_y_norm = img_y / (self.img_size - 1)
        img_x_norm = img_x / (self.img_size - 1)
        
        # 映射到传感器网格索引
        sensor_y = np.round(img_y_norm * (self.ny - 1)).astype(int)
        sensor_x = np.round(img_x_norm * (self.nx - 1)).astype(int)
        
        # 去重
        damaged_sensors = list(set(zip(sensor_y, sensor_x)))
        
        return damaged_sensors
    
    def crop_signal(
        self,
        signal: np.ndarray,
        damage_image: np.ndarray,
        return_grid: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
        """
        根据损伤位置裁剪信号
        
        Args:
            signal: (ny, nx, sig_len) 或 (batch, ny, nx, sig_len)
            damage_image: (img_size, img_size) 或 (batch, img_size, img_size)
            return_grid: 是否返回网格格式（True用于CNN, False用于DeepONet）
        
        Returns:
            cropped_signal: 裁剪后信号（用0填充被移除位置）
            kept_indices: 实际保留的传感器索引
            mask: (ny, nx) 二值掩码（1=保留，0=移除）
        """
        is_batch = signal.ndim == 4
        
        if not is_batch:
            signal = signal[np.newaxis, ...]  # (1, ny, nx, sig_len)
            damage_image = damage_image[np.newaxis, ...]  # (1, img_size, img_size)
        
        batch_size = signal.shape[0]
        
        # 初始化输出
        cropped_signals = signal.copy()  # 复制原始信号
        all_kept_indices = []
        all_masks = []
        
        for b in range(batch_size):
            sig = signal[b]  # (ny, nx, sig_len)
            dmg_img = damage_image[b]  # (img_size, img_size)
            
            # 1. 找出损伤对应的传感器
            damaged_sensors = self._map_damage_to_sensors(dmg_img)
            
            # 2. 生成所有传感器索引
            all_sensors = [(y, x) for y in range(self.ny) for x in range(self.nx)]
            
            # 3. 移除损伤传感器
            kept_sensors = [s for s in all_sensors if s not in damaged_sensors]
            
            # 4. 检查最小保留数
            if len(kept_sensors) < self.min_keep:
                # 如果移除太多，随机恢复一些
                need_restore = self.min_keep - len(kept_sensors)
                restore_sensors = random.sample(damaged_sensors, min(need_restore, len(damaged_sensors)))
                kept_sensors.extend(restore_sensors)
            
            # 5. 生成掩码
            mask = np.zeros((self.ny, self.nx), dtype=np.float32)
            for y, x in kept_sensors:
                mask[y, x] = 1.0
            
            # 6. 应用掩码（移除的位置填充0）
            for y in range(self.ny):
                for x in range(self.nx):
                    if mask[y, x] == 0:
                        cropped_signals[b, y, x, :] = 0.0
            
            all_kept_indices.append(kept_sensors)
            all_masks.append(mask)
        
        # 7. 处理返回格式
        if not return_grid:
            # Flatten用于DeepONet: (batch, ny*nx, sig_len)
            cropped_signals = cropped_signals.reshape(batch_size, -1, self.sig_len)
        
        if not is_batch:
            cropped_signals = cropped_signals[0]
            all_kept_indices = all_kept_indices[0]
            all_masks = all_masks[0]
        else:
            all_masks = np.stack(all_masks, axis=0)
        
        return cropped_signals, all_kept_indices, all_masks
    
    def visualize_damage_mapping(
        self,
        damage_image: np.ndarray,
        save_path: str = 'images/damage_mapping.png'
    ):
        """
        可视化损伤到传感器的映射关系
        
        Args:
            damage_image: (img_size, img_size)
            save_path: 保存路径
        """
        damaged_sensors = self._map_damage_to_sensors(damage_image)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：损伤概率图
        ax1 = axes[0]
        im1 = ax1.imshow(damage_image, cmap='hot', vmin=0, vmax=1, origin='lower')
        ax1.set_title('Damage Probability Map\n(10×10)', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label='Probability')
        
        # 右图：传感器掩码
        ax2 = axes[1]
        mask = np.ones((self.ny, self.nx))
        for y, x in damaged_sensors:
            mask[y, x] = 0
        
        im2 = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
        ax2.set_title(f'Sensor Mask (5×5)\nRemoved: {len(damaged_sensors)}/25', fontweight='bold')
        
        # 标记传感器
        for y in range(self.ny):
            for x in range(self.nx):
                if mask[y, x] == 1:
                    ax2.plot(x, y, 'go', markersize=15)
                    ax2.text(x, y, '✓', ha='center', va='center',
                            color='white', fontweight='bold')
                else:
                    ax2.plot(x, y, 'rx', markersize=15, markeredgewidth=3)
                    ax2.text(x, y, '✗', ha='center', va='center',
                            color='darkred', fontweight='bold')
        
        ax2.set_xticks(range(self.nx))
        ax2.set_yticks(range(self.ny))
        ax2.set_xlabel('x index')
        ax2.set_ylabel('y index')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Kept (1) / Removed (0)')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Damage mapping visualization saved to {save_path}")
        plt.close()


class DamageAwareCroppedDataset:
    """
    基于损伤的裁剪数据集包装器
    """
    
    def __init__(
        self,
        base_dataset,
        cropper: DamageAwareCropper,
        for_cnn: bool = True
    ):
        """
        Args:
            base_dataset: 原始数据集
            cropper: 损伤感知裁剪器
            for_cnn: 是否为CNN准备（保持网格格式）
        """
        self.base_dataset = base_dataset
        self.cropper = cropper
        self.for_cnn = for_cnn
        
        print(f"✓ DamageAwareCroppedDataset created:")
        print(f"  - Base dataset: {len(base_dataset)} samples")
        print(f"  - For CNN: {for_cnn}")
        print(f"  - Adaptive cropping based on damage location")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            cropped_signal: 裁剪后信号（0填充）
            image: 完整损伤图
        """
        signal, image = self.base_dataset[idx]
        
        # 根据损伤图裁剪信号
        cropped_signal, _, _ = self.cropper.crop_signal(
            signal,
            image,
            return_grid=self.for_cnn
        )
        
        return cropped_signal, image
    
    def get_branch_dim(self):
        """返回branch维度（用于DeepONet）"""
        return self.cropper.ny * self.cropper.nx * self.cropper.sig_len


class SubgridCropper:
    """
    子网格裁剪器 - 从大网格中提取子区域
    
    用途：从10×10传感器网格中提取中心5×5区域用于训练
    """
    
    def __init__(
        self,
        full_nx: int = 10,
        full_ny: int = 10,
        sub_nx: int = 5,
        sub_ny: int = 5,
        sig_len: int = 100,
        img_size: int = 10,
        position: str = 'center',  # 'center', 'corner'
        random_seed: Optional[int] = None
    ):
        """
        Args:
            full_nx, full_ny: 完整网格尺寸
            sub_nx, sub_ny: 子网格尺寸
            sig_len: 时间步长
            img_size: 损伤图尺寸
            position: 子网格位置（center-中心，corner-左上角）
            random_seed: 随机种子
        """
        assert sub_nx <= full_nx and sub_ny <= full_ny, "Subgrid must fit in full grid"
        
        self.full_nx = full_nx
        self.full_ny = full_ny
        self.sub_nx = sub_nx
        self.sub_ny = sub_ny
        self.sig_len = sig_len
        self.img_size = img_size
        self.position = position
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 计算子网格的起始索引
        if position == 'center':
            self.start_y = (full_ny - sub_ny) // 2
            self.start_x = (full_nx - sub_nx) // 2
        elif position == 'corner':
            self.start_y = 0
            self.start_x = 0
        else:
            raise ValueError(f"Unknown position: {position}")
        
        self.end_y = self.start_y + sub_ny
        self.end_x = self.start_x + sub_nx
        
        print(f"✓ SubgridCropper initialized:")
        print(f"  - Full grid: {full_ny}×{full_nx}")
        print(f"  - Sub grid: {sub_ny}×{sub_nx}")
        print(f"  - Position: {position}")
        print(f"  - Crop region: y[{self.start_y}:{self.end_y}], x[{self.start_x}:{self.end_x}]")
        print(f"  - Retention: {sub_nx*sub_ny}/{full_nx*full_ny} = {sub_nx*sub_ny/(full_nx*full_ny)*100:.1f}%")
    
    def crop_signal(
        self,
        signal: np.ndarray,
        return_grid: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        从完整信号中提取子区域
        
        Args:
            signal: (full_ny, full_nx, sig_len)
            return_grid: 是否保持网格格式
        
        Returns:
            cropped_signal: (sub_ny, sub_nx, sig_len) 或 (sub_ny*sub_nx, sig_len)
            kept_indices: 保留的传感器在原网格中的索引
        """
        # 提取子区域
        sub_signal = signal[self.start_y:self.end_y, self.start_x:self.end_x, :]
        
        # 记录保留的索引
        kept_indices = [
            (y, x) 
            for y in range(self.start_y, self.end_y) 
            for x in range(self.start_x, self.end_x)
        ]
        
        if not return_grid:
            # Flatten为一维
            sub_signal = sub_signal.reshape(-1, self.sig_len)
        
        return sub_signal, kept_indices
    
    def crop_image(
        self, 
        image: np.ndarray,
        target_size: Optional[int] = None  # 【新增】目标裁剪尺寸
    ) -> np.ndarray:
        """
        从完整损伤图中提取对应的子区域
        
        Args:
            image: (img_size, img_size) 完整损伤图
            target_size: 目标损伤图尺寸（None=保持原尺寸，int=裁剪到指定尺寸）
        
        Returns:
            sub_image: 裁剪后的损伤图
        """
        img_size = image.shape[0]
        
        # 计算对应的损伤图区域（传感器网格 → 损伤图坐标映射）
        img_start_y = int(self.start_y / self.full_ny * img_size)
        img_end_y = int(self.end_y / self.full_ny * img_size)
        img_start_x = int(self.start_x / self.full_nx * img_size)
        img_end_x = int(self.end_x / self.full_nx * img_size)
        
        if target_size is None:
            # 【修改】保持原尺寸，周围填0
            sub_image = np.zeros_like(image)
            sub_image[img_start_y:img_end_y, img_start_x:img_end_x] = \
                image[img_start_y:img_end_y, img_start_x:img_end_x]
        else:
            # 【新增】直接提取对应区域，resize到目标尺寸
            cropped_region = image[img_start_y:img_end_y, img_start_x:img_end_x]
            
            if cropped_region.shape[0] == target_size:
                sub_image = cropped_region
            else:
                # Resize到目标尺寸
                from scipy.ndimage import zoom
                scale = target_size / cropped_region.shape[0]
                sub_image = zoom(cropped_region, scale, order=1)
        
        return sub_image
    
    def visualize_crop_pattern(self) -> np.ndarray:
        """
        可视化裁剪模式
        
        Returns:
            mask: (full_ny, full_nx) 二值矩阵
        """
        mask = np.zeros((self.full_ny, self.full_nx), dtype=int)
        mask[self.start_y:self.end_y, self.start_x:self.end_x] = 1
        return mask


class SubgridCroppedDataset:
    """
    子网格裁剪数据集包装器
    """
    
    def __init__(
        self,
        base_dataset,
        cropper: SubgridCropper,
        for_cnn: bool = True,
        crop_target: bool = True,
        target_img_size: Optional[int] = None  # 【新增】目标损伤图尺寸
    ):
        """
        Args:
            base_dataset: 原始数据集（需要是10×10网格）
            cropper: 子网格裁剪器
            for_cnn: 是否为CNN准备（保持网格格式）
            crop_target: 是否裁剪目标损伤图（训练时True，测试时False）
            target_img_size: 目标损伤图尺寸（训练时=10，测试时=20）
        """
        self.base_dataset = base_dataset
        self.cropper = cropper
        self.for_cnn = for_cnn
        self.crop_target = crop_target
        self.target_img_size = target_img_size
        
        print(f"✓ SubgridCroppedDataset created:")
        print(f"  - Base dataset: {len(base_dataset)} samples")
        print(f"  - For CNN: {for_cnn}")
        print(f"  - Crop target: {crop_target}")
        if target_img_size is not None:
            print(f"  - Target image size: {target_img_size}×{target_img_size}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            cropped_signal: 裁剪后信号
            cropped_image: 裁剪后损伤图（或完整损伤图）
        """
        signal, image = self.base_dataset[idx]
        
        # 裁剪信号
        cropped_signal, _ = self.cropper.crop_signal(
            signal,
            return_grid=self.for_cnn
        )
        
        # 根据配置决定是否裁剪目标
        if self.crop_target:
            cropped_image = self.cropper.crop_image(image, target_size=self.target_img_size)
        else:
            cropped_image = image
        
        return cropped_signal, cropped_image
    
    def get_branch_dim(self):
        """返回branch维度（用于DeepONet）"""
        return self.cropper.sub_ny * self.cropper.sub_nx * self.cropper.sig_len
    
    def get_input_size(self):
        """返回CNN输入尺寸"""
        return self.cropper.sub_nx  # 假设nx=ny


# ==================== 工具函数 ====================

def create_cropped_dataset(
    base_dataset,
    crop_mode: str = 'boundary',
    n_keep: Optional[int] = None,
    random_per_sample: bool = True,
    random_seed: Optional[int] = None
):
    """
    便捷函数：创建裁剪数据集
    
    Args:
        base_dataset: 原始数据集
        crop_mode: 'boundary' 或 'random'
        n_keep: random模式下保留的传感器数量
        random_per_sample: 是否每个样本独立随机
        random_seed: 随机种子
    
    Returns:
        CroppedDatasetWrapper实例
    """
    # 从数据集获取参数
    info = base_dataset.get_info()
    ny, nx, sig_len = info['signal_shape']
    
    # 创建裁剪器
    cropper = SpatialCropper(
        nx=nx,
        ny=ny,
        sig_len=sig_len,
        crop_mode=crop_mode,
        n_keep=n_keep,
        random_seed=random_seed
    )
    
    # 创建包装器
    wrapped_dataset = CroppedDatasetWrapper(
        base_dataset,
        cropper,
        random_per_sample=random_per_sample
    )
    
    return wrapped_dataset, cropper

def create_square_cropped_dataset(
    base_dataset,
    crop_size: int = 3,
    crop_position: str = 'boundary',
    for_cnn: bool = True,
    random_seed: Optional[int] = None
):
    """
    便捷函数：创建正方形裁剪数据集
    
    Args:
        base_dataset: 原始数据集
        crop_size: 裁剪后尺寸（如3表示3×3）
        crop_position: 'center', 'corner', 'boundary', 'random'
        for_cnn: 是否为CNN准备（保持网格格式）
        random_seed: 随机种子
    
    Returns:
        SquareCroppedDatasetWrapper实例, SquareCropper实例
    """
    # 从数据集获取参数
    info = base_dataset.get_info()
    ny, nx, sig_len = info['signal_shape']
    
    # 创建裁剪器
    cropper = SquareCropper(
        nx=nx,
        ny=ny,
        sig_len=sig_len,
        crop_size=crop_size,
        crop_position=crop_position,
        random_seed=random_seed
    )
    
    # 创建包装器
    wrapped_dataset = SquareCroppedDatasetWrapper(
        base_dataset,
        cropper,
        for_cnn=for_cnn
    )
    
    return wrapped_dataset, cropper

def create_damage_aware_dataset(
    base_dataset,
    damage_threshold: float = 0.3,
    min_keep: int = 4,
    for_cnn: bool = True,
    random_seed: Optional[int] = None
):
    """
    便捷函数：创建基于损伤的裁剪数据集
    
    Args:
        base_dataset: 原始数据集
        damage_threshold: 损伤阈值
        min_keep: 最少保留传感器数
        for_cnn: 是否为CNN准备
        random_seed: 随机种子
    
    Returns:
        DamageAwareCroppedDataset实例, DamageAwareCropper实例
    """
    info = base_dataset.get_info()
    ny, nx, sig_len = info['signal_shape']
    img_size = info['image_shape'][0]
    
    cropper = DamageAwareCropper(
        nx=nx,
        ny=ny,
        sig_len=sig_len,
        img_size=img_size,
        damage_threshold=damage_threshold,
        min_keep=min_keep,
        random_seed=random_seed
    )
    
    wrapped_dataset = DamageAwareCroppedDataset(
        base_dataset,
        cropper,
        for_cnn=for_cnn
    )
    
    return wrapped_dataset, cropper

def create_subgrid_dataset(
    base_dataset,
    sub_nx: int = 5,
    sub_ny: int = 5,
    position: str = 'center',
    for_cnn: bool = True,
    crop_target: bool = True,
    target_img_size: Optional[int] = None,  # 【新增】
    random_seed: Optional[int] = None
):
    """
    便捷函数：创建子网格裁剪数据集
    
    Args:
        base_dataset: 原始数据集（需要是10×10网格）
        sub_nx, sub_ny: 子网格尺寸
        position: 裁剪位置
        for_cnn: 是否为CNN准备
        crop_target: 是否裁剪目标
        target_img_size: 目标损伤图尺寸（训练时=10，测试时=20）
        random_seed: 随机种子
    
    Returns:
        SubgridCroppedDataset实例, SubgridCropper实例
    """
    info = base_dataset.get_info()
    full_ny, full_nx, sig_len = info['signal_shape']
    img_size = info['image_shape'][0]
    
    # 验证数据集是否为10×10
    if full_nx != 10 or full_ny != 10:
        raise ValueError(f"SubgridCropper requires 10×10 base dataset, got {full_ny}×{full_nx}")
    
    cropper = SubgridCropper(
        full_nx=full_nx,
        full_ny=full_ny,
        sub_nx=sub_nx,
        sub_ny=sub_ny,
        sig_len=sig_len,
        img_size=img_size,
        position=position,
        random_seed=random_seed
    )
    
    wrapped_dataset = SubgridCroppedDataset(
        base_dataset,
        cropper,
        for_cnn=for_cnn,
        crop_target=crop_target,
        target_img_size=target_img_size  # 【新增】
    )
    
    return wrapped_dataset, cropper
