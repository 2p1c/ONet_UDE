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
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from data.dataset_simple import SimpleUSDataset3D
from data.transform import (
    create_cropped_dataset, 
    create_square_cropped_dataset, 
    create_damage_aware_dataset, 
    create_subgrid_dataset,  # 【新增】
    SquareCropper, 
    DamageAwareCropper
)
from nn.deeponet import DeepONet
from utils.data_utils import prepare_dataloaders
from utils.train_utils import train_model
from utils.visualization import (
    plot_loss_curves, 
    visualize_prediction,
    visualize_subgrid_training_flow  # 【新增】
)


def visualize_cropped_dataset_deeponet(raw_dataset, cropper, sample_idx=0, save_path='images/deeponet_cropped_data.png'):
    """
    可视化DeepONet使用的裁剪数据集
    
    展示内容：
    1. 原始 5×5 空间分布（某时刻）
    2. 裁剪掩码（哪些传感器被保留）
    3. 裁剪后信号的时域波形（flatten成一维）
    4. 目标损伤图（不变）
    
    Args:
        raw_dataset: 原始数据集
        cropper: 裁剪器（SquareCropper 或 DamageAwareCropper）
        sample_idx: 样本索引
        save_path: 保存路径
    """
    print(f"\n📊 Visualizing cropped dataset for DeepONet...")
    
    # 获取原始样本
    sig_full, img_target = raw_dataset[sample_idx]  # (5, 5, 100), (10, 10)
    
    # 【新增】根据裁剪器类型调用不同方法
    if isinstance(cropper, DamageAwareCropper):
        # 损伤感知裁剪需要传入损伤图
        sig_cropped, kept_indices, mask = cropper.crop_signal(
            sig_full,
            img_target,
            return_grid=False
        )
        # mask: (5, 5) 二值矩阵
        crop_mode_name = 'damage_aware'
    elif isinstance(cropper, SquareCropper):
        # 正方形裁剪
        sig_cropped, kept_indices = cropper.crop_signal(sig_full, return_grid=False)
        mask = cropper.visualize_crop_pattern()
        crop_mode_name = cropper.crop_position
    else:
        raise TypeError(f"Unsupported cropper type: {type(cropper)}")
    
    # 创建图形
    fig = plt.figure(figsize=(20, 10))
    
    # ===== 1. 原始 5×5 空间分布（某时刻）=====
    ax1 = plt.subplot(2, 4, 1)
    time_idx = 20
    spatial_full = sig_full[:, :, time_idx]
    spatial_full_interp = zoom(spatial_full, 8, order=1)
    
    im1 = ax1.imshow(spatial_full_interp, cmap='seismic',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal', vmin=-1, vmax=1)
    ax1.set_title('① Full Signal (5×5)\nat t=20μs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 标记所有传感器
    x_pos = np.linspace(0, 100, 5)
    y_pos = np.linspace(0, 100, 5)
    xv, yv = np.meshgrid(x_pos, y_pos)
    ax1.plot(xv.flatten(), yv.flatten(), 'ko', markersize=8, alpha=0.6)
    
    # ===== 2. 裁剪掩码 =====
    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    ax2.set_title(f'② Crop Mask\n({crop_mode_name} mode)', fontsize=12, fontweight='bold')
    
    for y in range(5):
        for x in range(5):
            if mask[y, x] == 1:
                ax2.plot(x, y, 'go', markersize=20)
                ax2.text(x, y, '✓', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=14)
            else:
                ax2.plot(x, y, 'rx', markersize=15, markeredgewidth=3)
                ax2.text(x, y, '✗', ha='center', va='center',
                        color='darkred', fontweight='bold', fontsize=14)
    
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xlabel('x index')
    ax2.set_ylabel('y index')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Kept (1) / Removed (0)')
    
    # ===== 3. 裁剪后空间分布 =====
    ax3 = plt.subplot(2, 4, 3)
    
    # 【修改】根据裁剪器类型重构信号
    if isinstance(cropper, DamageAwareCropper):
        # 损伤感知：直接使用原始5×5，被移除位置已为0
        spatial_cropped = sig_full[:, :, time_idx] * mask
        spatial_cropped_interp = zoom(spatial_cropped, 8, order=1)
        title_text = '③ Cropped Signal (5×5)\n(Removed = 0)'
    else:
        # 正方形裁剪：重构到3×3
        sig_cropped_grid = sig_cropped.reshape(3, 3, 100)
        spatial_cropped = sig_cropped_grid[:, :, time_idx]
        spatial_cropped_interp = zoom(spatial_cropped, 8, order=1)
        title_text = '③ Cropped Signal (3×3)\nat t=20μs'
    
    im3 = ax3.imshow(spatial_cropped_interp, cmap='seismic',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal', vmin=-1, vmax=1)
    ax3.set_title(title_text, fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 标记保留的传感器
    if isinstance(cropper, DamageAwareCropper):
        for y, x in kept_indices:
            x_mm = x * 25
            y_mm = y * 25
            ax3.plot(x_mm, y_mm, 'go', markersize=8)
    else:
        x_kept = np.linspace(0, 100, 3)
        y_kept = np.linspace(0, 100, 3)
        xv_kept, yv_kept = np.meshgrid(x_kept, y_kept)
        ax3.plot(xv_kept.flatten(), yv_kept.flatten(), 'go', markersize=8)
    
    # ===== 4. 目标损伤图（不变）=====
    ax4 = plt.subplot(2, 4, 4)
    im4 = ax4.imshow(img_target, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    ax4.set_title('④ Target (10×10)\n(Unchanged)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='Probability')
    
    # ===== 5. 完整信号时域波形（选3个点）=====
    ax5 = plt.subplot(2, 4, 5)
    t_vec = np.linspace(0, 100, 100)
    
    # 选择3个保留点的波形
    for i in range(min(3, len(kept_indices))):
        y_idx, x_idx = kept_indices[i]
        sig_point = sig_full[y_idx, x_idx, :]
        ax5.plot(t_vec, sig_point, linewidth=1.2, 
                label=f'Kept point ({x_idx},{y_idx})', alpha=0.8)
    
    ax5.set_xlabel('Time (μs)', fontsize=10)
    ax5.set_ylabel('Amplitude', fontsize=10)
    ax5.set_title('⑤ Time Signals (Kept Points)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    ax5.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    
    # ===== 6. 被移除点的波形（对比）=====
    ax6 = plt.subplot(2, 4, 6)
    
    # 找出被移除的点
    all_points = [(y, x) for y in range(5) for x in range(5)]
    removed_indices = [pt for pt in all_points if pt not in kept_indices]
    
    for i in range(min(3, len(removed_indices))):
        y_idx, x_idx = removed_indices[i]
        sig_point = sig_full[y_idx, x_idx, :]
        ax6.plot(t_vec, sig_point, linewidth=1.2, linestyle='--',
                label=f'Removed ({x_idx},{y_idx})', alpha=0.7)
    
    ax6.set_xlabel('Time (μs)', fontsize=10)
    ax6.set_ylabel('Amplitude', fontsize=10)
    ax6.set_title('⑥ Time Signals (Removed Points)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    ax6.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
    
    # ===== 7. DeepONet输入维度说明 =====
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    # 【修改】根据裁剪类型显示不同信息
    n_kept = len(kept_indices)
    n_total = 25
    retention_rate = n_kept / n_total * 100
    
    if isinstance(cropper, DamageAwareCropper):
        text_info = f"""
    📊 DeepONet Input (Damage-Aware)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Original Signal:
       • Shape: (5, 5, 100)
       • Total: 2500 time samples
       • Branch input: 2500 dims
    
    ✂️ After Cropping:
       • Shape: (5, 5, 100) → flatten
       • Kept: {n_kept}/{n_total} sensors
       • Branch input: 2500 dims (0-padded)
    
    📉 Effective Retention:
       {retention_rate:.1f}% sensors active
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🎯 Target (Unchanged):
       • Shape: (10, 10)
       • Total: 100 output points
    
    💡 Key Insight:
       Damaged sensors removed!
       DeepONet must infer from
       partial observations.
    """
    else:
        text_info = f"""
    📊 DeepONet Input Transformation
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Original Signal:
       • Shape: (5, 5, 100)
       • Total: 2500 time samples
       • Branch input: 2500 dims
    
    ✂️ After Cropping:
       • Shape: (3, 3, 100) → flatten
       • Total: 900 time samples
       • Branch input: 900 dims
    
    📉 Dimension Reduction:
       2500 → 900 ({retention_rate:.1f}% retained)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🎯 Target (Unchanged):
       • Shape: (10, 10)
       • Total: 100 output points
    
    💡 Key Insight:
       DeepONet must infer full damage
       from incomplete observations!
    """
    
    ax7.text(0.05, 0.5, text_info, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    # ===== 8. 信号能量分布对比 =====
    ax8 = plt.subplot(2, 4, 8)
    
    # 计算每个点的RMS能量
    rms_full = np.sqrt(np.mean(sig_full**2, axis=2))  # (5, 5)
    
    # 绘制热力图对比
    rms_full_interp = zoom(rms_full, 4, order=1)
    im8 = ax8.imshow(rms_full_interp, cmap='viridis',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    
    # 【修改】根据裁剪类型叠加不同的标记
    if isinstance(cropper, DamageAwareCropper):
        # 标记被移除的传感器
        for y, x in removed_indices:
            x_mm = x * 25
            y_mm = y * 25
            ax8.plot(x_mm, y_mm, 'rx', markersize=20, markeredgewidth=3)
        legend_text = 'Removed sensors'
    else:
        # 叠加裁剪区域框（中心3×3）
        rect_x = [20, 80, 80, 20, 20]
        rect_y = [20, 20, 80, 80, 20]
        ax8.plot(rect_x, rect_y, 'r-', linewidth=3)
        legend_text = 'Kept region'
    
    ax8.set_title('⑧ RMS Energy Distribution\n(Full grid)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('x (mm)')
    ax8.set_ylabel('y (mm)')
    plt.colorbar(im8, ax=ax8, shrink=0.8, label='RMS Energy')
    ax8.legend([legend_text], fontsize=9)
    
    plt.suptitle(f'DeepONet Cropped Dataset Visualization (Sample {sample_idx})\n'
                 f'Crop Mode: {crop_mode_name} | Kept: {n_kept}/{n_total} sensors ({retention_rate:.1f}%)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()


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
        default='square',
        choices=['boundary', 'random', 'square', 'damage_aware'],  # 【新增】
        help='裁剪模式：boundary-边界点, random-随机点, square-正方形裁剪, damage_aware-基于损伤裁剪'
    )
    parser.add_argument(
        '--crop-position',
        type=str,
        default='center',
        choices=['center', 'corner', 'boundary', 'random'],
        help='square模式下的裁剪位置：center-中心3×3, boundary-边界分散'
    )
    parser.add_argument(
        '--n-keep',
        type=int,
        default=None,
        help='random模式下保留的传感器数量'
    )
    parser.add_argument(
        '--damage-threshold',
        type=float,
        default=0.3,
        help='damage_aware模式下的损伤阈值'
    )
    parser.add_argument(
        '--min-keep',
        type=int,
        default=4,
        help='damage_aware模式下最少保留的传感器数'
    )
    parser.add_argument(
        '--use-subgrid',
        action='store_true',
        help='使用子网格训练模式（10×10→5×5）'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=10,
        help='损伤图尺寸（10或20）'
    )
    parser.add_argument(
        '--defect-range-full',
        action='store_true',
        help='损伤可出现在整个区域[0,1]（默认[0.2,0.8]）'
    )
    parser.add_argument(
        '--no-crop-input',
        action='store_true',
        help='【新增】使用完整传感器网格输入（不裁剪输入，只裁剪监督目标）'
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
    
    # 【修改】根据是否使用子网格，调整网格尺寸
    grid_nx = 10 if args.use_subgrid else config['nx']
    grid_ny = 10 if args.use_subgrid else config['ny']
    
    # 【修改】根据参数调整损伤范围
    defect_range = (0.0, 1.0) if args.defect_range_full else (0.2, 0.8)
    
    # 创建原始数据集
    raw_dataset = SimpleUSDataset3D(
        n_samples=config['n_samples'],
        nx=grid_nx,
        ny=grid_ny,
        sig_len=config['sig_len'],
        img_size=args.img_size,  # 【修改】支持20×20
        defect_range=defect_range,  # 【新增】
        precompute=True
    )
    
    print(f"✓ Base dataset loaded: {len(raw_dataset)} samples")
    print(f"  - Image size: {args.img_size}×{args.img_size}")
    print(f"  - Defect range: {defect_range}")
    
    # 【修改】子网格模式
    if args.use_subgrid:
        print(f"\n🔪 Applying subgrid crop (10×10 → 5×5)...")
        
        # 【新增】判断是否裁剪输入
        if args.no_crop_input:
            print(f"  ⚠️  Input NOT cropped (using full 10×10 signals)")
            print(f"  ✓  Target cropped to center 10×10 for supervision")
            
            # 创建特殊的数据集包装器：不裁剪输入，只裁剪目标
            from data.transform import SubgridCropper
            
            cropper = SubgridCropper(
                full_nx=10,
                full_ny=10,
                sub_nx=10,  # 【关键】不裁剪输入，保持10×10
                sub_ny=10,
                sig_len=config['sig_len'],
                img_size=args.img_size,
                position='center',
                random_seed=42
            )
            
            # 创建数据集：输入不裁剪，目标裁剪到10×10
            class FullInputSubgridDataset:
                """完整输入+裁剪目标的数据集"""
                def __init__(self, base_dataset, cropper, target_img_size=10):
                    self.base_dataset = base_dataset
                    self.cropper = cropper
                    self.target_img_size = target_img_size
                
                def __len__(self):
                    return len(self.base_dataset)
                
                def __getitem__(self, idx):
                    signal, image = self.base_dataset[idx]
                    # 输入：完整10×10信号（不裁剪）
                    # 目标：裁剪到中心10×10
                    cropped_target = self.cropper.crop_image(image, target_size=self.target_img_size)
                    return signal, cropped_target
                
                def get_branch_dim(self):
                    # 完整10×10信号展平
                    return 10 * 10 * self.cropper.sig_len
            
            dataset = FullInputSubgridDataset(raw_dataset, cropper, target_img_size=10)
            config['branch_dim'] = dataset.get_branch_dim()
            config['use_subgrid'] = True
            config['no_crop_input'] = True
            config['img_size'] = args.img_size
            config['train_img_size'] = 10
            
            print(f"✓ Full-input subgrid dataset created")
            print(f"  - Input: 10×10 signals (10000 dims)")
            print(f"  - Training supervision: 10×10 (center)")
            print(f"  - Full damage map: {args.img_size}×{args.img_size}")
        else:
            # 原有的裁剪输入逻辑
            dataset, cropper = create_subgrid_dataset(
                raw_dataset,
                sub_nx=5,
                sub_ny=5,
                position='center',
                for_cnn=False,
                crop_target=True,
                target_img_size=10,
                random_seed=42
            )
            config['branch_dim'] = dataset.get_branch_dim()
            config['use_subgrid'] = True
            config['no_crop_input'] = False  # 【新增】
            config['img_size'] = args.img_size
            config['train_img_size'] = 10
            print(f"✓ Subgrid dataset created")
            print(f"  - Training supervision: 10×10 (center)")
            print(f"  - Full damage map: {args.img_size}×{args.img_size}")
    elif config['use_crop']:
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
            
            # 【新增】可视化裁剪数据集
            print("\n🎨 Generating cropped dataset visualization...")
            visualize_cropped_dataset_deeponet(
                raw_dataset, 
                cropper, 
                sample_idx=0,
                save_path='images/deeponet_cropped_data_check.png'
            )
        elif config['crop_mode'] == 'damage_aware':
            # 【新增】基于损伤的裁剪
            dataset, cropper = create_damage_aware_dataset(
                raw_dataset,
                damage_threshold=args.damage_threshold,
                min_keep=args.min_keep,
                for_cnn=False,  # DeepONet需要flatten
                random_seed=42
            )
            config['branch_dim'] = dataset.get_branch_dim()
            config['damage_threshold'] = args.damage_threshold
            config['min_keep'] = args.min_keep
            
            # 可视化损伤映射
            print("\n🎨 Generating damage mapping visualization...")
            sample_sig, sample_img = raw_dataset[0]
            cropper.visualize_damage_mapping(
                sample_img,
                save_path='images/dataset_check/damage_mapping.png'
            )
            print(f"✓ Damage-aware cropped dataset created")
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
    
    # 【修复】无裁剪和子网格时的处理
    else:
        dataset = raw_dataset
        cropper = None
        config['branch_dim'] = config['nx'] * config['ny'] * config['sig_len']
        config['use_subgrid'] = False
        config['no_crop_input'] = False
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
    
    # 【修复】加载最佳模型并预测
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    visualize_prediction(
        model, 
        raw_dataset, 
        test_indices[0], 
        device,
        cropper=cropper
    )
    
    # 【新增】如果使用子网格模式，生成综合流程图
    if args.use_subgrid and args.img_size == 20:
        print("\n🎨 Generating comprehensive subgrid training flow...")
        visualize_subgrid_training_flow(
            model,
            raw_dataset,
            cropper,
            sample_idx=test_indices[0],
            device=device,
            no_crop_input=config.get('no_crop_input', False),  # 【新增】传递标志
            save_path='images/subgrid_training_flow.png'
        )
    
    # 【新增】如果使用了裁剪，再次可视化裁剪数据集（使用测试样本）
    if config['use_crop'] and cropper is not None:
        os.makedirs('images/dataset_check', exist_ok=True)
        print("\n🎨 Generating final cropped dataset visualization (test sample)...")
        visualize_cropped_dataset_deeponet(
            raw_dataset,
            cropper,
            sample_idx=test_indices[0],
            save_path='images/dataset_check/deeponet_cropped_test_sample.png'
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
