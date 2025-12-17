"""
CNN裁剪模式测试 - 验证训练/测试维度一致性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from data.dataset_simple import SimpleUSDataset3D
from data.transform import create_square_cropped_dataset
from nn.cnn import SimpleCNN
from utils.data_utils import prepare_cnn_dataloaders


def test_cnn_crop_consistency():
    """测试CNN裁剪训练的维度一致性"""
    print("=" * 70)
    print("Test: CNN Cropped Training - Dimension Consistency")
    print("=" * 70)
    
    # 1. 创建原始数据集
    raw_dataset = SimpleUSDataset3D(
        n_samples=100,
        nx=5, ny=5, sig_len=100,
        img_size=10,
        precompute=True
    )
    print(f"✓ Raw dataset: {len(raw_dataset)} samples")
    
    # 2. 创建裁剪数据集
    cropped_dataset, cropper = create_square_cropped_dataset(
        raw_dataset,
        crop_size=3,
        crop_position='center',
        for_cnn=True,  # CNN模式
        random_seed=42
    )
    print(f"✓ Cropped dataset: {len(cropped_dataset)} samples")
    
    # 3. 检查单个样本形状
    sig_cropped, img = cropped_dataset[0]
    print(f"\n✓ Cropped sample shapes:")
    print(f"  - Signal: {sig_cropped.shape} (expected: (3, 3, 100))")
    print(f"  - Image: {img.shape} (expected: (10, 10))")
    
    assert sig_cropped.shape == (3, 3, 100), f"Shape mismatch: {sig_cropped.shape}"
    assert img.shape == (10, 10), f"Shape mismatch: {img.shape}"
    
    # 4. 创建数据加载器
    train_loader, test_loader, _, test_indices = prepare_cnn_dataloaders(
        cropped_dataset,
        train_ratio=0.8,
        batch_size=16
    )
    
    # 5. 检查batch形状
    data_batch, target_batch = next(iter(train_loader))
    print(f"\n✓ Training batch shapes:")
    print(f"  - Data: {data_batch.shape} (expected: (16, 100, 3, 3))")
    print(f"  - Target: {target_batch.shape} (expected: (16, 1, 10, 10))")
    
    assert data_batch.shape == (16, 100, 3, 3), f"Batch shape error: {data_batch.shape}"
    
    # 6. 创建模型并测试前向传播
    model = SimpleCNN(
        input_channels=100,
        hidden_channels=64,
        dropout=0.0,
        input_size=3  # ← 关键：输入尺寸为3
    )
    
    print(f"\n✓ Model initialized:")
    model_info = model.get_info()
    print(f"  - Input: {model_info['input_shape']}")
    print(f"  - Output: {model_info['output_shape']}")
    
    # 7. 测试前向传播
    model.eval()
    with torch.no_grad():
        output = model(data_batch)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Output shape: {output.shape} (expected: (16, 1, 10, 10))")
    
    assert output.shape == (16, 1, 10, 10), f"Output shape error: {output.shape}"
    
    # 8. 测试可视化流程
    print(f"\n✓ Testing visualization flow...")
    
    # 从原始数据集获取样本
    sig_full, img_true = raw_dataset[test_indices[0]]
    print(f"  - Full signal: {sig_full.shape}")
    
    # 裁剪信号
    sig_vis, _ = cropper.crop_signal(sig_full, return_grid=True)
    print(f"  - Cropped for vis: {sig_vis.shape}")
    
    assert sig_vis.shape == (3, 3, 100), f"Vis shape error: {sig_vis.shape}"
    
    # 转换为CNN输入
    sig_cnn = np.transpose(sig_vis, (2, 0, 1))
    sig_tensor = torch.from_numpy(sig_cnn).unsqueeze(0)
    print(f"  - CNN tensor: {sig_tensor.shape} (expected: (1, 100, 3, 3))")
    
    assert sig_tensor.shape == (1, 100, 3, 3), f"Tensor shape error: {sig_tensor.shape}"
    
    # 预测
    with torch.no_grad():
        pred = model(sig_tensor)
    
    print(f"  - Prediction: {pred.shape} (expected: (1, 1, 10, 10))")
    assert pred.shape == (1, 1, 10, 10), f"Pred shape error: {pred.shape}"
    
    img_pred = pred.squeeze().numpy()
    print(f"  - Output image: {img_pred.shape} (expected: (10, 10))")
    
    # 9. 统计信息
    print(f"\n✓ Prediction statistics:")
    print(f"  - Min: {img_pred.min():.6f}")
    print(f"  - Max: {img_pred.max():.6f}")
    print(f"  - Mean: {img_pred.mean():.6f}")
    print(f"  - In [0,1]: {np.all((img_pred >= 0) & (img_pred <= 1))}")
    
    print("\n" + "=" * 70)
    print("✅ All dimension consistency tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_cnn_crop_consistency()
