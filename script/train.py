"""
DeepONet训练脚本 - 精简版

数据流程: 5×5×50信号 → DeepONet → 10×10损伤图

【新增】支持裁剪数据集训练，验证泛化能力
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import json
import argparse  # 新增

from data.dataset_simple import SimpleUSDataset3D
from data.transform import create_cropped_dataset, create_square_cropped_dataset  # 修改导入
from nn.deeponet import DeepONet
from utils.data_utils import prepare_dataloaders
from utils.train_utils import train_model
from utils.visualization import plot_loss_curves, visualize_prediction


def main():
    """主训练流程"""
    # ==================== 解析命令行参数 ====================
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument(
        '--crop',
        action='store_true',
        help='使用裁剪数据集训练'
    )
    parser.add_argument(
        '--crop-mode',
        type=str,
        default='square',  # 【修改】默认square模式
        choices=['boundary', 'random', 'square'],  # 【新增】square选项
        help='裁剪模式：boundary-边界点, random-随机点, square-正方形裁剪'
    )
    parser.add_argument(
        '--crop-position',
        type=str,
        default='boundary',
        choices=['center', 'corner', 'boundary', 'random'],
        help='square模式下的裁剪位置'
    )
    parser.add_argument(
        '--n-keep',
        type=int,
        default=None,
        help='random模式下保留的传感器数量'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepONet Training - Simplified")
    if args.crop:
        print(f"【裁剪模式】Mode: {args.crop_mode}")
    print("=" * 70)
    
    # ==================== 配置参数 ====================
    config = {
        # 数据参数
        'n_samples': 2000,
        'train_ratio': 0.8,
        'nx': 5,
        'ny': 5,
        'sig_len': 100,
        # 【关键】branch_dim 将根据是否裁剪自动调整
        'branch_dim': None,  # 稍后设置
        'trunk_dim': 2,
        'branch_depth': 2,
        'trunk_depth': 3,
        'width': 100,
        'dropout': 0.15,
        # 训练参数
        'batch_size': 128,
        'epochs': 100,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        # 早停参数
        'early_stopping': True,
        'patience': 20,
        # 学习率调度
        'use_scheduler': True,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        # 【新增】裁剪参数
        'use_crop': args.crop,
        'crop_mode': args.crop_mode,
        'n_keep': args.n_keep,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # ==================== 加载数据 ====================
    print("\n" + "=" * 70)
    print("Loading Dataset...")
    print("=" * 70)
    
    # 创建原始数据集
    raw_dataset = SimpleUSDataset3D(
        n_samples=config['n_samples'],
        nx=config['nx'],
        ny=config['ny'],
        sig_len=config['sig_len'],
        img_size=10,
        precompute=True
    )
    
    print(f"✓ Base dataset loaded: {len(raw_dataset)} samples")
    
    # 【新增】根据是否裁剪，包装数据集
    cropper = None
    if config['use_crop']:
        print(f"\n🔪 Applying crop transform...")
        
        if config['crop_mode'] == 'square':
            # 【新增】正方形裁剪（3×3）
            dataset, cropper = create_square_cropped_dataset(
                raw_dataset,
                crop_size=3,
                crop_position=args.crop_position,
                for_cnn=False,  # DeepONet需要flatten
                random_seed=42
            )
            config['branch_dim'] = dataset.get_branch_dim()
        else:
            # 原有的boundary/random模式
            dataset, cropper = create_cropped_dataset(
                raw_dataset,
                crop_mode=config['crop_mode'],
                n_keep=config['n_keep'],
                random_per_sample=True,
                random_seed=42
            )
            config['branch_dim'] = dataset.get_branch_dim()
        
        print(f"✓ Cropped dataset created")
    else:
        dataset = raw_dataset
        config['branch_dim'] = config['nx'] * config['ny'] * config['sig_len']
        print(f"✓ Using full dataset (no crop)")
    
    print(f"\n✓ Training config:")
    print(f"  - Samples: {config['n_samples']}")
    print(f"  - Spatial grid: {config['nx']}×{config['ny']}")
    print(f"  - Time steps: {config['sig_len']}")
    print(f"  - Branch dim: {config['branch_dim']}")
    if config['use_crop']:
        print(f"  - Crop mode: {config['crop_mode']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['lr']}")
    
    # ==================== 准备数据加载器 ====================
    train_loader, test_loader, train_indices, test_indices = prepare_dataloaders(
        dataset,
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
        dropout=config.get('dropout', 0.0)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {n_params:,} parameters")
    
    # ==================== 训练 ====================
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    train_losses, test_losses, best_loss = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, device,
        epochs=config['epochs'],
        save_path='checkpoints/best_model.pth',
        early_stopping=config['early_stopping'],
        patience=config['patience'],
        use_scheduler=config['use_scheduler'],
        scheduler_patience=config['scheduler_patience'],
        scheduler_factor=config['scheduler_factor']
    )
    
    # ==================== 保存配置和指标 ====================
    print("\nSaving training configuration and metrics...")
    
    config_save_path = 'checkpoints/last_config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to: {config_save_path}")
    
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
    
    plot_loss_curves(train_losses, test_losses)
    
    # 【修复】加载最佳模型并预测，传递cropper
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    visualize_prediction(
        model, 
        raw_dataset, 
        test_indices[0], 
        device,
        cropper=cropper  # 【新增】传递裁剪器
    )
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"✓ Epochs: {config['epochs']}")
    print(f"✓ Final train loss: {train_losses[-1]:.6f}")
    print(f"✓ Final test loss: {test_losses[-1]:.6f}")
    print(f"✓ Best test loss: {best_loss:.6f}")
    if config['use_crop']:
        print(f"✓ Training mode: Cropped ({config['crop_mode']})")
    else:
        print(f"✓ Training mode: Full")
    print("=" * 70)
    print("🎉 Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
