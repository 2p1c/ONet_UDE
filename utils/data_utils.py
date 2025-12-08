"""
数据处理工具模块
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


class DeepONetDataset(Dataset):
    """
    DeepONet专用数据集 - 将原始数据扩展为(branch, trunk, target)格式
    
    每个原始样本(sig, img)生成100条训练数据 (10×10个位置)
    """
    
    def __init__(self, raw_dataset, sample_indices: List[int], verbose: bool = True):
        """
        Args:
            raw_dataset: SimpleUSDataset3D实例
            sample_indices: 用于训练/测试的样本索引列表
            verbose: 是否打印详细信息
        """
        self.raw_dataset = raw_dataset
        self.sample_indices = sample_indices
        
        # 预处理: 扩展数据
        self.branch_inputs = []
        self.trunk_inputs = []
        self.targets = []
        
        if verbose:
            print(f"Expanding {len(sample_indices)} samples to training data...")
        
        for count, idx in enumerate(sample_indices, 1):
            sig, img = raw_dataset[idx]
            
            # Branch输入: 展平信号 (5, 5, 50) -> (1250,)
            branch_vec = sig.flatten()
            
            # 遍历损伤图的每个位置
            img_size = img.shape[0]
            for i in range(img_size):
                for j in range(img_size):
                    # Trunk输入: 归一化坐标
                    x_norm = j / (img_size - 1)
                    y_norm = i / (img_size - 1)
                    trunk_vec = np.array([x_norm, y_norm], dtype=np.float32)
                    
                    # 目标: 损伤概率值
                    target_val = img[i, j]
                    
                    self.branch_inputs.append(branch_vec)
                    self.trunk_inputs.append(trunk_vec)
                    self.targets.append(target_val)
            
            # 进度显示
            if verbose and (count % 5 == 0 or count == len(sample_indices)):
                print(f"  Progress: {count}/{len(sample_indices)} samples processed")
        
        # 转换为numpy数组
        self.branch_inputs = np.array(self.branch_inputs, dtype=np.float32)
        self.trunk_inputs = np.array(self.trunk_inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32).reshape(-1, 1)
        
        if verbose:
            print(f"✓ Generated {len(self)} training samples")
            print(f"  - Branch: {self.branch_inputs.shape}")
            print(f"  - Trunk: {self.trunk_inputs.shape}")
            print(f"  - Target: {self.targets.shape}")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        """Returns: x (branch+trunk), y (target)"""
        branch = self.branch_inputs[idx]
        trunk = self.trunk_inputs[idx]
        x = np.concatenate([branch, trunk])
        y = self.targets[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def prepare_dataloaders(
    raw_dataset,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """
    准备训练和测试数据加载器
    
    Args:
        raw_dataset: 原始数据集
        train_ratio: 训练集比例
        batch_size: 批大小
        seed: 随机种子
    
    Returns:
        train_loader, test_loader, train_indices, test_indices
    """
    n_samples = len(raw_dataset)
    n_train = int(n_samples * train_ratio)
    
    # 划分训练/测试集
    all_indices = list(range(n_samples))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    
    train_indices = all_indices[:n_train]
    test_indices = all_indices[n_train:]
    
    print(f"\n✓ Data split:")
    print(f"  - Train: {len(train_indices)} samples → {len(train_indices) * 100} points")
    print(f"  - Test: {len(test_indices)} samples → {len(test_indices) * 100} points")
    
    # 构建DeepONet数据集
    train_dataset = DeepONetDataset(raw_dataset, train_indices)
    test_dataset = DeepONetDataset(raw_dataset, test_indices, verbose=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_indices, test_indices
