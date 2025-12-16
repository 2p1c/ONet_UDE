"""
CNN训练脚本 - 基于2D卷积网络

数据流程: (5×5×100)信号 → CNN → (10×10)损伤图

架构说明:
- 输入重塑: (5, 5, 100) → (100, 5, 5)  # 时间作为通道
- CNN提取特征并上采样到(10, 10)
- 输出Sigmoid激活保证[0,1]概率值
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import json
import argparse  # 新增

from data.dataset_simple import SimpleUSDataset3D
from data.transform import create_square_cropped_dataset  # 新增
from nn.cnn import SimpleCNN
from utils.data_utils import prepare_cnn_dataloaders
from utils.train_utils import train_model
from utils.visualization import plot_loss_curves, visualize_prediction


def visualize_cnn_prediction(model, raw_dataset, sample_idx, device, save_path='images/cnn_prediction.png'):
    """
    CNN专用可视化函数
    
    Args:
        model: CNN模型
        raw_dataset: 原始数据集
        sample_idx: 样本索引
        device: 设备
        save_path: 保存路径
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    model.eval()
    
    # 获取原始样本
    sig, img_true = raw_dataset[sample_idx]  # (5, 5, 100), (10, 10)
    
    # 转换为CNN输入格式
    sig_cnn = np.transpose(sig, (2, 0, 1))  # (100, 5, 5)
    sig_tensor = torch.from_numpy(sig_cnn).unsqueeze(0).to(device)  # (1, 100, 5, 5)
    
    # 预测
    with torch.no_grad():
        pred = model(sig_tensor)  # (1, 1, 10, 10)
        img_pred = pred.squeeze().cpu().numpy()  # (10, 10)
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 真实损伤图
    im0 = axes[0].imshow(img_true, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='Probability')
    
    # 预测损伤图
    im1 = axes[1].imshow(img_pred, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[1].set_title('CNN Prediction', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], label='Probability')
    
    # 误差图
    error = np.abs(img_pred - img_true)
    im2 = axes[2].imshow(error, cmap='viridis', vmin=0, vmax=0.5, origin='lower')
    axes[2].set_title(f'Absolute Error (MAE={error.mean():.4f})', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2], label='|Error|')
    
    plt.suptitle(f'CNN Prediction (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Prediction visualization saved to {save_path}")
    plt.close()


def main():
    """主训练流程"""
    # ==================== 解析命令行参数 ====================
    parser = argparse.ArgumentParser(description='CNN Training')
    parser.add_argument(
        '--crop',
        action='store_true',
        help='使用裁剪数据集训练（3×3网格）'
    )
    parser.add_argument(
        '--crop-position',
        type=str,
        default='boundary',
        choices=['center', 'corner', 'boundary', 'random'],
        help='裁剪位置'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("CNN Training - Simple 2D Convolutional Network")
    if args.crop:
        print(f"【裁剪模式】Position: {args.crop_position}")
    print("=" * 70)
    
    # ==================== 配置参数 ====================
    config = {
        # 数据参数（与train.py保持一致）
        'n_samples': 2000,
        'train_ratio': 0.8,
        'nx': 5,
        'ny': 5,
        'sig_len': 100,  # 时间步长
        'img_size': 10,
        # CNN网络参数
        'input_channels': 100,  # 与sig_len一致
        'hidden_channels': 64,
        'dropout': 0.15,
        # 训练参数（参考train.py）
        'batch_size': 128,
        'epochs': 100,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        # 早停和学习率调度
        'early_stopping': True,
        'patience': 20,
        'use_scheduler': True,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    print(f"✓ Training config:")
    print(f"  - Samples: {config['n_samples']}")
    print(f"  - Spatial grid: {config['nx']}×{config['ny']}")
    print(f"  - Time steps: {config['sig_len']}")
    print(f"  - Image size: {config['img_size']}×{config['img_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['lr']}")
    print(f"  - Dropout: {config['dropout']}")
    
    # ==================== 加载数据 ====================
    print("\n" + "=" * 70)
    print("Loading Dataset...")
    print("=" * 70)
    
    raw_dataset = SimpleUSDataset3D(
        n_samples=config['n_samples'],
        nx=config['nx'],
        ny=config['ny'],
        sig_len=config['sig_len'],
        img_size=config['img_size'],
        precompute=True
    )
    
    print(f"✓ Base dataset loaded: {len(raw_dataset)} samples")
    
    # 【新增】根据是否裁剪，包装数据集
    cropper = None
    input_size = config['nx']  # 默认5×5
    
    if args.crop:
        print(f"\n🔪 Applying square crop transform...")
        dataset, cropper = create_square_cropped_dataset(
            raw_dataset,
            crop_size=3,
            crop_position=args.crop_position,
            for_cnn=True,  # 保持网格格式
            random_seed=42
        )
        input_size = 3  # 裁剪后3×3
        print(f"✓ Cropped dataset created")
    else:
        dataset = raw_dataset
        print(f"✓ Using full dataset (no crop)")
    
    print(f"\n✓ Dataset info:")
    print(f"  - Samples: {len(dataset)}")
    print(f"  - Input size: {input_size}×{input_size}×{config['sig_len']}")
    print(f"  - Output size: {config['img_size']}×{config['img_size']}")
    
    # ==================== 准备数据加载器 ====================
    train_loader, test_loader, train_indices, test_indices = prepare_cnn_dataloaders(
        dataset,
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size']
    )
    
    # ==================== 初始化模型 ====================
    print("\n" + "=" * 70)
    print("Initializing Model...")
    print("=" * 70)
    
    model = SimpleCNN(
        input_channels=config['input_channels'],
        hidden_channels=config['hidden_channels'],
        dropout=config['dropout'],
        input_size=input_size  # 【新增】传入输入尺寸
    ).to(device)
    
    # 打印模型信息
    model_info = model.get_info()
    print(f"✓ Model: {model_info['model_name']}")
    print(f"  - Input: {model_info['input_shape']}")
    print(f"  - Output: {model_info['output_shape']}")
    print(f"  - Parameters: {model_info['total_parameters']:,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # ==================== 训练 ====================
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    train_losses, test_losses, best_loss = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, device,
        epochs=config['epochs'],
        save_path='checkpoints/best_cnn_model.pth',
        early_stopping=config['early_stopping'],
        patience=config['patience'],
        use_scheduler=config['use_scheduler'],
        scheduler_patience=config['scheduler_patience'],
        scheduler_factor=config['scheduler_factor']
    )
    
    # ==================== 保存配置和指标 ====================
    print("\nSaving training configuration and metrics...")
    
    # 保存配置
    config['use_crop'] = args.crop
    config['crop_position'] = args.crop_position if args.crop else None
    config['input_size'] = input_size
    
    config_save_path = 'checkpoints/last_cnn_config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to: {config_save_path}")
    
    # 保存训练指标
    metrics = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_loss,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'n_epochs_completed': len(train_losses),
    }
    metrics_save_path = 'checkpoints/last_cnn_metrics.pth'
    torch.save(metrics, metrics_save_path)
    print(f"✓ Metrics saved to: {metrics_save_path}")
    
    # ==================== 可视化 ====================
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)
    
    os.makedirs('images', exist_ok=True)
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, test_losses, save_path='images/cnn_loss_curve.png')
    
    # 加载最佳模型并预测
    model.load_state_dict(torch.load('checkpoints/best_cnn_model.pth'))
    visualize_cnn_prediction(model, raw_dataset, test_indices[0], device)
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"✓ Model: SimpleCNN")
    print(f"✓ Input size: {input_size}×{input_size}")
    print(f"✓ Epochs: {len(train_losses)}")
    print(f"✓ Final train loss: {train_losses[-1]:.6f}")
    print(f"✓ Final test loss: {test_losses[-1]:.6f}")
    print(f"✓ Best test loss: {best_loss:.6f}")
    if args.crop:
        print(f"✓ Training mode: Cropped ({args.crop_position})")
    else:
        print(f"✓ Training mode: Full")
    print("=" * 70)
    print("🎉 CNN Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
