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
