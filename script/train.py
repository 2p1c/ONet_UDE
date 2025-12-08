"""
DeepONet训练脚本 - 精简版

数据流程: 5×5×50信号 → DeepONet → 10×10损伤图
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import json  # 新增导入

from data.dataset_simple import SimpleUSDataset3D
from nn.deeponet import DeepONet
from utils.data_utils import prepare_dataloaders
from utils.train_utils import train_model
from utils.visualization import plot_loss_curves, visualize_prediction


def main():
    """主训练流程"""
    print("=" * 70)
    print("DeepONet Training - Simplified")
    print("=" * 70)
    
    # ==================== 配置参数 ====================
    config = {
        # 数据参数
        'n_samples': 2000,  # 【改进4】从 1000 增加到 2000
        'train_ratio': 0.8,
        # 【关键修复】网络参数 - branch_dim 必须与数据集时间步长一致
        'nx': 5,
        'ny': 5,
        'sig_len': 100,
        'branch_dim': 5 * 5 * 100,
        'trunk_dim': 2,
        'branch_depth': 2,
        'trunk_depth': 3,
        'width': 100,  # 【改进7】从 50 增加到 100
        'dropout': 0.15,  # 【改进3】添加 Dropout
        # 训练参数
        'batch_size': 128,  # 【改进5】从 64 增加到 128
        'epochs': 100,
        'lr': 5e-4,  # 【改进1】从 1e-3 降到 5e-4
        # 【新增】正则化参数
        'weight_decay': 1e-4,
        # 【新增】早停参数
        'early_stopping': True,
        'patience': 20,  # 【优化】从 15 增加到 20
        # 【新增】学习率调度参数
        'use_scheduler': True,
        'scheduler_patience': 5,  # 【改进2】从 10 改为 5
        'scheduler_factor': 0.5,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    print(f"✓ Training config:")
    print(f"  - Samples: {config['n_samples']}")
    print(f"  - Spatial grid: {config['nx']}×{config['ny']}")
    print(f"  - Time steps: {config['sig_len']}")
    print(f"  - Branch dim: {config['branch_dim']}")  # 新增显示
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['lr']}")
    print(f"  - Weight decay: {config['weight_decay']}")
    print(f"  - Early stopping: patience={config['patience']}")
    
    # ==================== 加载数据 ====================
    print("\n" + "=" * 70)
    print("Loading Dataset...")
    print("=" * 70)
    
    # 【修复】使用配置中的参数创建数据集
    raw_dataset = SimpleUSDataset3D(
        n_samples=config['n_samples'],
        nx=config['nx'],
        ny=config['ny'],
        sig_len=config['sig_len'],  # 使用配置中的时间步长
        img_size=10,
        precompute=True
    )
    
    print(f"✓ Dataset loaded: {len(raw_dataset)} samples")
    print(f"  - Signal shape: ({config['ny']}, {config['nx']}, {config['sig_len']})")
    
    # ==================== 准备数据加载器 ====================
    train_loader, test_loader, train_indices, test_indices = prepare_dataloaders(
        raw_dataset,
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size']
    )
    
    # ==================== 初始化模型 ====================
    print("\n" + "=" * 70)
    print("Initializing Model...")
    print("=" * 70)
    
    model = DeepONet(
        branch_dim=config['branch_dim'],
        trunk_dim=config['trunk_dim'],
        branch_depth=config['branch_depth'],
        trunk_depth=config['trunk_depth'],
        width=config['width'],
        activation='relu',
        initializer='Glorot normal',
        dropout=config.get('dropout', 0.0)  # 【新增】传入 Dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    
    # 【修改】添加权重衰减（L2正则化）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']  # L2正则化
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {n_params:,} parameters")
    
    # ==================== 训练 ====================
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # 【修改】传入新参数
    train_losses, test_losses, best_loss = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, device,
        epochs=config['epochs'],
        save_path='checkpoints/best_model.pth',
        # 【新增】早停和学习率调度
        early_stopping=config['early_stopping'],
        patience=config['patience'],
        use_scheduler=config['use_scheduler'],
        scheduler_patience=config['scheduler_patience'],
        scheduler_factor=config['scheduler_factor']
    )
    
    # 【新增】保存训练配置和指标
    print("\nSaving training configuration and metrics...")
    
    # 保存配置
    config_save_path = 'checkpoints/last_config.json'
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
    metrics_save_path = 'checkpoints/last_metrics.pth'
    torch.save(metrics, metrics_save_path)
    print(f"✓ Metrics saved to: {metrics_save_path}")
    
    # ==================== 可视化 ====================
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)
    
    os.makedirs('images', exist_ok=True)
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, test_losses)
    
    # 加载最佳模型并预测
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    visualize_prediction(model, raw_dataset, test_indices[0], device)
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"✓ Epochs: {config['epochs']}")
    print(f"✓ Final train loss: {train_losses[-1]:.6f}")
    print(f"✓ Final test loss: {test_losses[-1]:.6f}")
    print(f"✓ Best test loss: {best_loss:.6f}")
    print("=" * 70)
    print("🎉 Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
