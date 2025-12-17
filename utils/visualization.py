"""
可视化工具模块 - 集中管理所有可视化函数

功能分类：
1. 训练相关：损失曲线、预测结果
2. 数据集检查：原始数据、裁剪数据、对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import List
from scipy.ndimage import zoom


# ==================== 工具函数 ====================

def ensure_dir(path: str):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)


# ==================== 训练相关可视化 ====================

def plot_loss_curves(
    train_losses: List[float],
    test_losses: List[float],
    save_path: str = 'images/train_loss_curve.png'
):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Loss curve saved to {save_path}")
    plt.close()


def visualize_prediction(
    model,
    dataset,
    sample_idx: int,
    device,
    save_path: str = 'images/train_prediction.png',
    cropper=None
):
    """
    可视化DeepONet预测结果
    
    Args:
        model: 训练好的DeepONet模型
        dataset: 原始数据集
        sample_idx: 样本索引
        device: 设备
        save_path: 保存路径
        cropper: 裁剪器（可选，用于显示裁剪信息）
    """
    model.eval()
    
    # 获取原始数据
    sig_full, img_true = dataset[sample_idx]
    
    # 【修改】处理裁剪信号
    if cropper is not None:
        from data.transform import SquareCropper, DamageAwareCropper, SubgridCropper  # 【新增】SubgridCropper
        
        if isinstance(cropper, SquareCropper):
            # 正方形裁剪（flatten用于DeepONet）
            sig_cropped, kept_indices = cropper.crop_signal(sig_full, return_grid=False)
            sig_input = sig_cropped
        elif isinstance(cropper, DamageAwareCropper):
            # 损伤感知裁剪
            sig_cropped, kept_indices, mask = cropper.crop_signal(
                sig_full,
                img_true,
                return_grid=False  # DeepONet使用flatten
            )
            sig_input = sig_cropped
        elif isinstance(cropper, SubgridCropper):
            # 【新增】子网格裁剪
            sig_cropped, kept_indices = cropper.crop_signal(sig_full, return_grid=False)
            sig_input = sig_cropped
        else:
            raise TypeError(f"Unknown cropper type: {type(cropper)}")
    else:
        # 完整信号
        sig_input = sig_full.reshape(-1, sig_full.shape[-1])
    
    # Flatten信号
    sig_flat = sig_input.flatten()
    
    # 获取数据集信息
    info = dataset.get_info()
    img_size = info['image_shape'][0]
    
    # 构建预测网格
    x_grid = np.linspace(0, 1, img_size)
    y_grid = np.linspace(0, 1, img_size)
    xv, yv = np.meshgrid(x_grid, y_grid, indexing='xy')
    
    pred_img = np.zeros((img_size, img_size))
    
    # 逐点预测
    with torch.no_grad():
        for i in range(img_size):
            for j in range(img_size):
                trunk_input = np.array([xv[i, j], yv[i, j]], dtype=np.float32)
                x_input = np.concatenate([sig_flat, trunk_input])
                x_input = torch.from_numpy(x_input).unsqueeze(0).to(device)
                
                pred_val = model(x_input).cpu().numpy()[0, 0]
                pred_img[i, j] = pred_val
    
    # 归一化预测结果到 [0, 1]
    pred_img = np.clip(pred_img, 0, 1)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 真实标签
    ax1 = axes[0]
    im1 = ax1.imshow(img_true, cmap='hot', vmin=0, vmax=1, origin='lower')
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 预测结果
    ax2 = axes[1]
    im2 = ax2.imshow(pred_img, cmap='hot', vmin=0, vmax=1, origin='lower')
    ax2.set_title('Prediction', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 误差图
    ax3 = axes[2]
    error = np.abs(pred_img - img_true)
    im3 = ax3.imshow(error, cmap='coolwarm', vmin=0, vmax=0.5, origin='lower')
    mae = error.mean()
    ax3.set_title(f'Absolute Error (MAE={mae:.4f})', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 标题添加模式信息
    mode_str = "Cropped Input" if cropper is not None else "Full Input"
    plt.suptitle(f'DeepONet Prediction - Sample {sample_idx} ({mode_str})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存
    save_path = 'images/deeponet_prediction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Prediction visualization saved to {save_path}")
    plt.close()
    
    # 打印误差统计
    rmse = np.sqrt(np.mean(error**2))
    print(f"\n✓ Prediction metrics:")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - Max error: {error.max():.6f}")


def visualize_cnn_prediction(
    model,
    dataset,
    sample_idx: int,
    device,
    save_path: str = 'images/cnn_prediction.png',
    cropper=None
):
    """
    可视化CNN预测结果
    
    Args:
        model: 训练好的CNN模型
        dataset: 原始数据集
        sample_idx: 样本索引
        device: 设备
        save_path: 保存路径
        cropper: 裁剪器（可选）
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from data.transform import SquareCropper, DamageAwareCropper, SubgridCropper  # 【新增】SubgridCropper
    
    model.eval()
    
    # 获取原始数据
    sig_full, img_target = dataset[sample_idx]
    
    # 【修改】根据裁剪器类型处理输入
    if cropper is not None:
        if isinstance(cropper, SquareCropper):
            # 正方形裁剪：3×3网格
            sig_input, kept_indices = cropper.crop_signal(sig_full, return_grid=True)
            mask = cropper.visualize_crop_pattern()
            crop_mode = f"Square ({cropper.crop_position})"
            is_damage_aware = False
            is_subgrid = False
        elif isinstance(cropper, DamageAwareCropper):
            # 损伤感知裁剪：5×5网格（部分为0）
            sig_input, kept_indices, mask = cropper.crop_signal(
                sig_full,
                img_target,
                return_grid=True  # CNN需要网格格式
            )
            crop_mode = "Damage-Aware"
            is_damage_aware = True
            is_subgrid = False
        elif isinstance(cropper, SubgridCropper):
            # 【新增】子网格裁剪：从10×10提取5×5
            sig_input, kept_indices = cropper.crop_signal(sig_full, return_grid=True)
            mask = cropper.visualize_crop_pattern()
            crop_mode = f"Subgrid ({cropper.position})"
            is_damage_aware = False
            is_subgrid = True
        else:
            raise TypeError(f"Unsupported cropper type: {type(cropper)}")
    else:
        # 无裁剪：完整5×5
        sig_input = sig_full
        mask = np.ones((5, 5))
        kept_indices = [(y, x) for y in range(5) for x in range(5)]
        crop_mode = "Full"
        is_damage_aware = False
        is_subgrid = False
    
    # 预测
    with torch.no_grad():
        # CNN输入: (batch, channels, height, width) = (1, 100, H, W)
        sig_tensor = torch.FloatTensor(sig_input).permute(2, 0, 1).unsqueeze(0).to(device)
        pred = model(sig_tensor).squeeze().cpu().numpy()
    
    # 可视化
    fig = plt.figure(figsize=(20, 10))
    
    # 1. 原始信号空间分布
    ax1 = plt.subplot(2, 4, 1)
    time_idx = 20
    spatial_full = sig_full[:, :, time_idx]
    spatial_full_interp = zoom(spatial_full, 8, order=1)
    im1 = ax1.imshow(spatial_full_interp, cmap='seismic', vmin=-1, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    
    # 【修改】根据是否子网格调整标题
    if is_subgrid:
        ax1.set_title(f'① Original Signal ({cropper.full_ny}×{cropper.full_nx})\nat t=20μs', fontweight='bold')
    else:
        ax1.set_title('① Original Signal\n(5×5 at t=20μs)', fontweight='bold')
    
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. 裁剪掩码
    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    ax2.set_title(f'② Crop Mask\n({crop_mode})', fontweight='bold')
    
    # 【修改】根据mask尺寸调整标记
    mask_ny, mask_nx = mask.shape
    for y in range(mask_ny):
        for x in range(mask_nx):
            if mask[y, x] == 1:
                ax2.plot(x, y, 'go', markersize=15)
            else:
                ax2.plot(x, y, 'rx', markersize=15, markeredgewidth=2)
    ax2.set_xticks(range(mask_nx))
    ax2.set_yticks(range(mask_ny))
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. 输入信号
    ax3 = plt.subplot(2, 4, 3)
    spatial_input = sig_input[:, :, time_idx]
    spatial_input_interp = zoom(spatial_input, 8, order=1)
    im3 = ax3.imshow(spatial_input_interp, cmap='seismic', vmin=-1, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    
    if is_damage_aware:
        title_text = '③ CNN Input (5×5)\n(Removed = 0)'
    elif is_subgrid:
        title_text = f'③ CNN Input ({cropper.sub_ny}×{cropper.sub_nx})\n(Subgrid)'
    elif cropper is not None and not is_subgrid:
        title_text = '③ CNN Input (3×3)\n(Cropped)'
    else:
        title_text = '③ CNN Input (5×5)\n(Full)'
    
    ax3.set_title(title_text, fontweight='bold')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 4. 目标损伤图
    ax4 = plt.subplot(2, 4, 4)
    im4 = ax4.imshow(img_target, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    ax4.set_title('④ Target\n(10×10)', fontweight='bold')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # 5. 预测结果
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(pred, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower')
    ax5.set_title('⑤ Prediction\n(10×10)', fontweight='bold')
    ax5.set_xlabel('x (mm)')
    ax5.set_ylabel('y (mm)')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    # 6. 误差图
    ax6 = plt.subplot(2, 4, 6)
    error = np.abs(pred - img_target)
    im6 = ax6.imshow(error, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower')
    mae = error.mean()
    ax6.set_title(f'⑥ Absolute Error\nMAE={mae:.4f}', fontweight='bold')
    ax6.set_xlabel('x (mm)')
    ax6.set_ylabel('y (mm)')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    # 7. 时域波形（保留的传感器）
    ax7 = plt.subplot(2, 4, 7)
    t_vec = np.linspace(0, 100, sig_full.shape[-1])
    for i in range(min(3, len(kept_indices))):
        y, x = kept_indices[i]
        ax7.plot(t_vec, sig_full[y, x, :], linewidth=1.2,
                label=f'Sensor ({x},{y})', alpha=0.8)
    ax7.set_title('⑦ Time Signals\n(Kept sensors)', fontweight='bold')
    ax7.set_xlabel('Time (μs)')
    ax7.set_ylabel('Amplitude')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. 预测 vs 目标对比
    ax8 = plt.subplot(2, 4, 8)
    ax8.plot(img_target.flatten(), label='Target', linewidth=2, alpha=0.7)
    ax8.plot(pred.flatten(), label='Prediction', linewidth=2, alpha=0.7)
    ax8.set_title('⑧ Flatten Comparison', fontweight='bold')
    ax8.set_xlabel('Spatial index')
    ax8.set_ylabel('Probability')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 总标题
    n_kept = len(kept_indices)
    
    # 【修改】根据裁剪类型计算retention
    if is_subgrid:
        n_total = cropper.full_nx * cropper.full_ny
    else:
        n_total = 25
    
    retention = n_kept / n_total * 100
    
    plt.suptitle(f'CNN Prediction Visualization (Sample {sample_idx})\n'
                 f'Mode: {crop_mode} | Sensors: {n_kept}/{n_total} ({retention:.1f}%) | MAE: {mae:.4f}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ CNN prediction visualization saved to {save_path}")
    plt.close()


# ==================== 数据集检查可视化 ====================

def visualize_simple_dataset(dataset, sample_idx=0, save_path='images/dataset_check/simple_dataset.png'):
    """
    可视化简化数据集
    
    Args:
        dataset: SimpleUSDataset3D 实例
        sample_idx: 样本索引
        save_path: 保存路径（默认保存到 dataset_check/ 文件夹）
    """
    print("=" * 60)
    print("Visualization: Simple Dataset Analysis")
    print("=" * 60)
    
    sig, img = dataset[sample_idx]
    print(f"Data shapes: signal{sig.shape}, image{img.shape}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 5))
    
    # ===== 1. 某个点的时域波形 =====
    ax1 = plt.subplot(1, 3, 1)
    sample_y, sample_x = 2, 2
    time_signal = sig[sample_y, sample_x, :]
    t_vec = np.linspace(0, dataset.T, dataset.sig_len)
    ax1.plot(t_vec * 1e6, time_signal, linewidth=1.2, color='steelblue')
    ax1.set_xlabel('Time (μs)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Time Signal at Point ({sample_x}, {sample_y})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # ===== 2. 空间分布 =====
    ax2 = plt.subplot(1, 3, 2)
    time_idx = 8
    spatial_snapshot = sig[:, :, time_idx]
    spatial_interp = zoom(spatial_snapshot, 4, order=1)
    
    im2 = ax2.imshow(spatial_interp, cmap='seismic',
                     extent=[0, dataset.L * 1000, 0, dataset.L * 1000],
                     origin='lower', aspect='equal', vmin=-1, vmax=1)
    ax2.set_xlabel('x (mm)', fontsize=11)
    ax2.set_ylabel('y (mm)', fontsize=11)
    ax2.set_title(f'Spatial Distribution at t={t_vec[time_idx]*1e6:.1f}μs', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Amplitude', shrink=0.8)
    
    ax2.plot(dataset.src_x * 1000, dataset.src_y * 1000, 'g*', markersize=15, label='Source')
    x_sensors = np.linspace(0, dataset.L, dataset.nx) * 1000
    y_sensors = np.linspace(0, dataset.L, dataset.ny) * 1000
    xv_s, yv_s = np.meshgrid(x_sensors, y_sensors)
    ax2.plot(xv_s.flatten(), yv_s.flatten(), 'ko', markersize=4, label='Sensors')
    ax2.legend(fontsize=9)
    
    # ===== 3. 损伤概率图 =====
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(img, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    ax3.set_xlabel('x (mm)', fontsize=11)
    ax3.set_ylabel('y (mm)', fontsize=11)
    ax3.set_title('Damage Probability Map', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Probability', shrink=0.8)
    
    plt.suptitle('Simple Dataset Visualization (5×5×50 → 10×10)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
    
    print("✅ Simple dataset visualization complete!\n")


def visualize_crop_pattern(cropper, save_path='images/dataset_check/crop_pattern.png'):
    """
    可视化裁剪模式
    
    Args:
        cropper: SpatialCropper 实例
        save_path: 保存路径
    """
    print("=" * 60)
    print("Visualization: Crop Pattern")
    print("=" * 60)
    
    mask = cropper.visualize_crop_pattern()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    im = ax.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
    crop_mode = getattr(cropper, 'crop_mode', 'unknown')
    n_keep = np.sum(mask)
    ax.set_title(f'Crop Pattern: {crop_mode} mode ({int(n_keep)} sensors)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    
    # 标注保留点
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == 1:
                ax.plot(x, y, 'go', markersize=15)
                ax.text(x, y, '✓', ha='center', va='center',
                       color='white', fontweight='bold', fontsize=12)
            else:
                ax.plot(x, y, 'rx', markersize=12, markeredgewidth=2)
    
    ax.set_xticks(range(mask.shape[1]))
    ax.set_yticks(range(mask.shape[0]))
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='Kept (1) / Removed (0)')
    
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
    
    print("✅ Crop pattern visualization complete!\n")


def visualize_cropped_sample_comparison(base_dataset, cropper, sample_idx=0, 
                                       save_path='images/dataset_check/cropped_comparison.png'):
    """
    可视化裁剪前后的信号对比
    
    Args:
        base_dataset: 原始数据集
        cropper: 裁剪器
        sample_idx: 样本索引
        save_path: 保存路径
    """
    print("=" * 60)
    print("Visualization: Before/After Cropping")
    print("=" * 60)
    
    sig_full, img = base_dataset[sample_idx]
    sig_cropped, kept_indices = cropper.crop_signal(sig_full, random_per_sample=False)
    
    fig = plt.figure(figsize=(18, 10))
    
    # === 1. 完整信号空间分布 ===
    ax1 = plt.subplot(2, 3, 1)
    time_idx = 20
    spatial_full = sig_full[:, :, time_idx]
    spatial_full_interp = zoom(spatial_full, 8, order=1)
    
    im1 = ax1.imshow(spatial_full_interp, cmap='seismic',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal', vmin=-1, vmax=1)
    ax1.set_title('Full Signal (5×5 sensors)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    x_pos = np.linspace(0, 100, 5)
    y_pos = np.linspace(0, 100, 5)
    xv, yv = np.meshgrid(x_pos, y_pos)
    ax1.plot(xv.flatten(), yv.flatten(), 'ko', markersize=6, label='All sensors')
    ax1.legend(fontsize=9)
    
    # === 2. 裁剪模式掩码 ===
    ax2 = plt.subplot(2, 3, 2)
    mask = cropper.visualize_crop_pattern()
    im2 = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title(f'Crop Pattern ({len(kept_indices)} sensors)', fontsize=12, fontweight='bold')
    
    for y in range(5):
        for x in range(5):
            if mask[y, x] == 1:
                ax2.plot(x, y, 'go', markersize=12)
            else:
                ax2.plot(x, y, 'rx', markersize=10)
    
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # === 3. 损伤图 ===
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(img, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    ax3.set_title('Target (unchanged)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # === 4. 完整信号时域波形 ===
    ax4 = plt.subplot(2, 3, 4)
    t_vec = np.linspace(0, 100, sig_full.shape[2])
    center_sig = sig_full[2, 2, :]
    ax4.plot(t_vec, center_sig, linewidth=1.2, label='Center (2,2)')
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Full Signal - Center Point', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # === 5. 裁剪信号时域波形 ===
    ax5 = plt.subplot(2, 3, 5)
    for i in range(min(4, len(kept_indices))):
        y_idx, x_idx = kept_indices[i]
        original_sig = sig_full[y_idx, x_idx, :]
        ax5.plot(t_vec, original_sig, linewidth=1.0,
                label=f'Kept ({x_idx},{y_idx})', alpha=0.8)
    
    ax5.set_xlabel('Time (μs)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('Cropped Signals - Kept Points', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    
    # === 6. 维度对比 ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    n_full = sig_full.shape[0] * sig_full.shape[1] * sig_full.shape[2]
    n_crop = sig_cropped.shape[0] * sig_cropped.shape[1]
    
    text_info = f"""
    📊 Dimension Comparison
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Full Dataset:
       • Signal shape: {sig_full.shape}
       • Flattened: {n_full} dims
    
    🔪 Cropped Dataset:
       • Signal shape: {sig_cropped.shape}
       • Flattened: {n_crop} dims
    
    📉 Dimension reduction:
       {n_full} → {n_crop}
       ({n_crop/n_full*100:.1f}% of original)
    
    ✅ Target unchanged:
       {img.shape} = {img.size} points
    """
    
    ax6.text(0.1, 0.5, text_info, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    
    plt.suptitle('Spatial Cropping: Input Reduction with Unchanged Target',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
    
    print("✅ Cropped sample visualization complete!\n")


def visualize_cropped_dataset_deeponet(raw_dataset, cropper, sample_idx=0, save_path='images/dataset_check/deeponet_cropped_data.png'):
    """
    可视化DeepONet使用的裁剪数据集
    
    展示内容：
    1. 原始 5×5 空间分布
    2. 裁剪掩码
    3. 裁剪后信号
    4. 目标损伤图
    """
    print(f"\n📊 Visualizing cropped dataset for DeepONet...")
    
    sig_full, img_target = raw_dataset[sample_idx]
    sig_cropped, kept_indices = cropper.crop_signal(sig_full, return_grid=False)
    mask = cropper.visualize_crop_pattern()
    
    fig = plt.figure(figsize=(20, 10))
    
    # 原始空间分布
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
    
    x_pos = np.linspace(0, 100, 5)
    y_pos = np.linspace(0, 100, 5)
    xv, yv = np.meshgrid(x_pos, y_pos)
    ax1.plot(xv.flatten(), yv.flatten(), 'ko', markersize=8, alpha=0.6)
    
    # 裁剪掩码
    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    ax2.set_title(f'② Crop Mask\n({cropper.crop_position} mode)', fontsize=12, fontweight='bold')
    
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
    
    # 裁剪后空间分布
    ax3 = plt.subplot(2, 4, 3)
    sig_cropped_grid = sig_cropped.reshape(3, 3, 100)
    spatial_cropped = sig_cropped_grid[:, :, time_idx]
    spatial_cropped_interp = zoom(spatial_cropped, 8, order=1)
    
    im3 = ax3.imshow(spatial_cropped_interp, cmap='seismic',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal', vmin=-1, vmax=1)
    ax3.set_title('③ Cropped Signal (3×3)\nat t=20μs', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    x_kept = np.linspace(0, 100, 3)
    y_kept = np.linspace(0, 100, 3)
    xv_kept, yv_kept = np.meshgrid(x_kept, y_kept)
    ax3.plot(xv_kept.flatten(), yv_kept.flatten(), 'go', markersize=8)
    
    # 目标损伤图
    ax4 = plt.subplot(2, 4, 4)
    im4 = ax4.imshow(img_target, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    ax4.set_title('④ Target (10×10)\n(Unchanged)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='Probability')
    
    # 完整信号时域波形
    ax5 = plt.subplot(2, 4, 5)
    t_vec = np.linspace(0, 100, 100)
    
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
    
    # 被移除点的波形
    ax6 = plt.subplot(2, 4, 6)
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
    
    # 维度说明
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    n_full = sig_full.shape[0] * sig_full.shape[1] * sig_full.shape[2]
    n_crop = sig_cropped.shape[0] * sig_cropped.shape[1]
    
    text_info = f"""
    📊 Dimension Comparison
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Full Dataset:
       • Signal shape: {sig_full.shape}
       • Flattened: {n_full} dims
    
    🔪 Cropped Dataset:
       • Signal shape: {sig_cropped.shape}
       • Flattened: {n_crop} dims
    
    📉 Dimension reduction:
       {n_full} → {n_crop}
       ({n_crop/n_full*100:.1f}% of original)
    
    ✅ Target unchanged:
       {img.shape} = {img.size} points
    """
    
    ax7.text(0.1, 0.5, text_info, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    plt.suptitle('Spatial Cropping: Input Reduction with Unchanged Target',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
    
    print("✅ Cropped sample visualization complete!\n")


def visualize_cropped_dataset_cnn(raw_dataset, cropper, sample_idx=0, save_path='images/dataset_check/cnn_cropped_data.png'):
    """可视化CNN使用的裁剪数据集"""
    from data.transform import SquareCropper, DamageAwareCropper
    
    print(f"\n📊 Visualizing cropped dataset for CNN...")
    
    sig_full, img_target = raw_dataset[sample_idx]
    
    # 【修改】根据裁剪器类型调用不同方法
    if isinstance(cropper, DamageAwareCropper):
        # 损伤感知裁剪需要传入损伤图
        sig_cropped, kept_indices, mask = cropper.crop_signal(
            sig_full,
            img_target,
            return_grid=True  # CNN需要网格格式
        )
        crop_mode_name = 'damage_aware'
    elif isinstance(cropper, SquareCropper):
        # 正方形裁剪
        sig_cropped, kept_indices = cropper.crop_signal(sig_full, return_grid=True)
        mask = cropper.visualize_crop_pattern()
        crop_mode_name = cropper.crop_position
    else:
        raise TypeError(f"Unsupported cropper type: {type(cropper)}")
    
    fig = plt.figure(figsize=(20, 10))
    
    # 原始空间分布
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
    
    x_pos = np.linspace(0, 100, 5)
    y_pos = np.linspace(0, 100, 5)
    xv, yv = np.meshgrid(x_pos, y_pos)
    ax1.plot(xv.flatten(), yv.flatten(), 'ko', markersize=8, alpha=0.6)
    
    # 裁剪掩码
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
    
    # 裁剪后空间分布
    ax3 = plt.subplot(2, 4, 3)
    
    # 【修改】根据裁剪器类型处理
    if isinstance(cropper, DamageAwareCropper):
        # 损伤感知：直接显示5×5（被移除位置为0）
        spatial_cropped = sig_cropped[:, :, time_idx]
        spatial_cropped_interp = zoom(spatial_cropped, 8, order=1)
        title_text = '③ CNN Input (5×5)\n(Removed = 0)'
    else:
        # 正方形裁剪：显示3×3
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
    
    # 目标损伤图
    ax4 = plt.subplot(2, 4, 4)
    im4 = ax4.imshow(img_target, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    ax4.set_title('④ Target (10×10)\n(Unchanged)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='Probability')
    
    # 完整信号时域波形
    ax5 = plt.subplot(2, 4, 5)
    t_vec = np.linspace(0, 100, 100)
    
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
    
    # 被移除点的波形
    ax6 = plt.subplot(2, 4, 6)
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
    
    # 维度说明
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    # 【修改】根据裁剪类型显示不同信息
    n_kept = len(kept_indices)
    n_total = 25
    retention_rate = n_kept / n_total * 100
    
    if isinstance(cropper, DamageAwareCropper):
        text_info = f"""
    📊 CNN Input (Damage-Aware)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Original Signal:
       • Shape: (5, 5, 100)
       • Grid: 5×5 sensors
       • Channels: 100 (time)
    
    ✂️ After Cropping:
       • Shape: (5, 5, 100) unchanged
       • Kept: {n_kept}/{n_total} sensors
       • Removed positions = 0
    
    📉 Effective Retention:
       {retention_rate:.1f}% sensors active
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🎯 Target (Unchanged):
       • Shape: (10, 10)
    
    💡 Key: CNN learns to ignore
       zero-padded positions!
    """
    else:
        text_info = f"""
    📊 CNN Input Transformation
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Original Signal:
       • Shape: (5, 5, 100)
       • Total: 2500 time samples
       • Input channels: 5
    
    ✂️ After Cropping:
       • Shape: (3, 3, 100)
       • Total: 900 time samples
       • Input channels: 3
    
    📉 Dimension Reduction:
       2500 → 900 ({retention_rate:.1f}% retained)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🎯 Target (Unchanged):
       • Shape: (10, 10)
    
    💡 CNN learns local features
       directly from spatial grid.
    """
    
    ax7.text(0.05, 0.5, text_info, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    # RMS能量分布
    ax8 = plt.subplot(2, 4, 8)
    rms_full = np.sqrt(np.mean(sig_full**2, axis=2))
    rms_full_interp = zoom(rms_full, 4, order=1)
    im8 = ax8.imshow(rms_full_interp, cmap='viridis',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    
    # 【修改】根据裁剪类型叠加不同标记
    if isinstance(cropper, DamageAwareCropper):
        # 标记被移除的传感器
        for y, x in removed_indices:
            x_mm = x * 25
            y_mm = y * 25
            ax8.plot(x_mm, y_mm, 'rx', markersize=20, markeredgewidth=3)
        legend_text = 'Removed sensors'
    else:
        # 叠加裁剪区域框
        rect_x = [20, 80, 80, 20, 20]
        rect_y = [20, 20, 80, 80, 20]
        ax8.plot(rect_x, rect_y, 'r-', linewidth=3)
        legend_text = 'Kept region'
    
    ax8.set_title('⑧ RMS Energy Distribution\n(Full grid)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('x (mm)')
    ax8.set_ylabel('y (mm)')
    plt.colorbar(im8, ax=ax8, shrink=0.8, label='RMS Energy')
    ax8.legend([legend_text], fontsize=9)
    
    plt.suptitle(f'CNN Cropped Dataset Visualization (Sample {sample_idx})\n'
                 f'Crop Mode: {crop_mode_name} | Kept: {n_kept}/{n_total} sensors ({retention_rate:.1f}%)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()


def visualize_subgrid_training_flow(
    model,
    raw_dataset,
    cropper,
    sample_idx: int,
    device,
    no_crop_input: bool = False,
    save_path: str = 'images/subgrid_training_flow.png'
):
    """可视化子网格训练的完整数据流程"""
    from scipy.ndimage import zoom
    import matplotlib.patches as mpatches
    
    model.eval()
    
    # ===== 1. 获取原始数据 =====
    sig_10x10, img_20x20_target = raw_dataset[sample_idx]  # (10,10,100), (20,20)
    
    # ===== 2. 裁剪到训练数据 =====
    if no_crop_input:
        # 【修复】完整输入模式：展平10×10×100→一维向量
        sig_input = sig_10x10.reshape(-1, sig_10x10.shape[-1])  # (100,100)
        sig_input = sig_input.flatten()  # 【关键修复】展平为(10000,)
        kept_indices = [(y, x) for y in range(10) for x in range(10)]
        img_10x10_train = cropper.crop_image(img_20x20_target, target_size=10)
        
        input_shape_str = "10×10×100"
        input_dim = 10000
    else:
        # 原有的5×5裁剪模式
        sig_5x5, kept_indices = cropper.crop_signal(sig_10x10, return_grid=False)
        sig_input = sig_5x5.flatten()  # 已经是(2500,)
        img_10x10_train = cropper.crop_image(img_20x20_target, target_size=10)
        
        input_shape_str = "5×5×100"
        input_dim = 2500
    
    # ===== 3. DeepONet预测（多分辨率查询）=====
    predictions = {}
    for query_size in [10, 20]:
        x_grid = np.linspace(0, 1, query_size)
        y_grid = np.linspace(0, 1, query_size)
        xv, yv = np.meshgrid(x_grid, y_grid, indexing='xy')
        
        pred_img = np.zeros((query_size, query_size))
        
        with torch.no_grad():
            for i in range(query_size):
                for j in range(query_size):
                    trunk_input = np.array([xv[i, j], yv[i, j]], dtype=np.float32)
                    # 【验证】sig_input应该是一维(10000,)或(2500,), trunk_input是(2,)
                    x_input = np.concatenate([sig_input, trunk_input])
                    x_tensor = torch.from_numpy(x_input).unsqueeze(0).to(device)
                    pred_val = model(x_tensor).cpu().numpy()[0, 0]
                    pred_img[i, j] = pred_val
        
        predictions[query_size] = np.clip(pred_img, 0, 1)
    
    # ===== 4. 绘制综合图 =====
    fig = plt.figure(figsize=(24, 16))
    
    # ========== 第一行：数据生成阶段 ==========
    
    # 1.1 完整10×10传感器网格信号（某时刻）
    ax1 = plt.subplot(3, 5, 1)
    time_idx = 20
    spatial_10x10 = sig_10x10[:, :, time_idx]
    spatial_10x10_interp = zoom(spatial_10x10, 5, order=1)
    im1 = ax1.imshow(spatial_10x10_interp, cmap='seismic', vmin=-1, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    ax1.set_title('① Full Sensor Grid\n10×10 @ t=20μs', fontweight='bold', fontsize=11)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 标记所有传感器
    x_pos_10 = np.linspace(0, 100, 10)
    y_pos_10 = np.linspace(0, 100, 10)
    xv_10, yv_10 = np.meshgrid(x_pos_10, y_pos_10)
    ax1.plot(xv_10.flatten(), yv_10.flatten(), 'ko', markersize=4, alpha=0.5)
    
    # 1.2 裁剪掩码
    ax2 = plt.subplot(3, 5, 2)
    mask_10x10 = cropper.visualize_crop_pattern()
    im2 = ax2.imshow(mask_10x10, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    
    if no_crop_input:
        ax2.set_title('② Crop Mask\n(No Crop - Full 10×10)', fontweight='bold', fontsize=11, color='green')
    else:
        ax2.set_title('② Crop Mask\n(Center 5×5)', fontweight='bold', fontsize=11)
    
    for y in range(10):
        for x in range(10):
            if mask_10x10[y, x] == 1:
                ax2.plot(x, y, 'go', markersize=8)
            else:
                ax2.plot(x, y, 'rx', markersize=6, markeredgewidth=1.5, alpha=0.5)
    
    ax2.set_xticks(range(0, 10, 2))
    ax2.set_yticks(range(0, 10, 2))
    ax2.grid(True, alpha=0.2)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 1.3 输入信号可视化
    ax3 = plt.subplot(3, 5, 3)
    if no_crop_input:
        # 【修复】完整10×10输入的可视化
        spatial_input = sig_10x10[:, :, time_idx]
        spatial_input_interp = zoom(spatial_input, 5, order=1)
        title_text = '③ Training Input\n10×10 Signal (Full)'
        marker_color = 'green'
    else:
        # 5×5裁剪输入
        sig_5x5_reshaped = sig_input.reshape(5, 5, 100)  # 【修复】从展平的恢复
        spatial_input = sig_5x5_reshaped[:, :, time_idx]
        spatial_input_interp = zoom(spatial_input, 10, order=1)
        title_text = '③ Training Input\n5×5 Signal'
        marker_color = 'blue'
    
    im3 = ax3.imshow(spatial_input_interp, cmap='seismic', vmin=-1, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    ax3.set_title(title_text, fontweight='bold', fontsize=11, color=marker_color)
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 1.4 完整20×20损伤图（真值）
    ax4 = plt.subplot(3, 5, 4)
    im4 = ax4.imshow(img_20x20_target, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    ax4.set_title('④ Full Target\n20×20 Damage Map', fontweight='bold', fontsize=11)
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # 标记训练监督区域（中心10×10）
    rect = mpatches.Rectangle((25, 25), 50, 50, linewidth=2.5,
                              edgecolor='cyan', facecolor='none',
                              linestyle='--', label='Supervised (10×10)')
    ax4.add_patch(rect)
    ax4.legend(fontsize=9)
    
    # 1.5 训练监督目标（中心10×10）
    ax5 = plt.subplot(3, 5, 5)
    im5 = ax5.imshow(img_10x10_train, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    ax5.set_title('⑤ Training Target\n10×10 (Center)', fontweight='bold', fontsize=11)
    ax5.set_xlabel('x (mm)')
    ax5.set_ylabel('y (mm)')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    # ========== 第二行：预测结果 ==========
    
    # 2.1 时域信号
    ax6 = plt.subplot(3, 5, 6)
    t_vec = np.linspace(0, 100, 100)
    for i in range(min(3, len(kept_indices))):
        y, x = kept_indices[i]
        ax6.plot(t_vec, sig_10x10[y, x, :], linewidth=1.2,
                label=f'Sensor ({x},{y})', alpha=0.8)
    ax6.set_title('⑥ Time Signals\n(Input Sensors)', fontweight='bold', fontsize=11)
    ax6.set_xlabel('Time (μs)')
    ax6.set_ylabel('Amplitude')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 2.2 DeepONet架构示意
    ax7 = plt.subplot(3, 5, 7)
    ax7.axis('off')
    arch_text = f"""
    🔷 DeepONet Architecture
    
    ━━━━━━━━━━━━━━━━━━━━━━━
    
    Branch Net:
    • Input: {input_shape_str} = {input_dim}
    • Output: 100 (basis)
    
    {'⚠️ Full Input (No Crop)' if no_crop_input else '✂️ Cropped Input (5×5)'}
    
    Trunk Net:
    • Input: (x, y) coords
    • Output: 100 (weights)
    
    Prediction:
    • G(u, y) = Σ bᵢ(u) ψᵢ(y)
    • Can query ANY (x,y)!
    
    ━━━━━━━━━━━━━━━━━━━━━━━
    
    Training:
    • Supervised: 10×10 center
    • Loss: MSE on center
    
    Testing:
    • Query 10×10: ✓ High acc
    • Query 20×20: ? Extrapolation
    """
    ax7.text(0.05, 0.5, arch_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    # 2.3 预测结果（10×10，训练分辨率）
    ax8 = plt.subplot(3, 5, 8)
    pred_10x10 = predictions[10]
    im8 = ax8.imshow(pred_10x10, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    ax8.set_title('⑧ Prediction (10×10)\nTrained Resolution', 
                  fontweight='bold', fontsize=11, color='green')
    ax8.set_xlabel('x (mm)')
    ax8.set_ylabel('y (mm)')
    plt.colorbar(im8, ax=ax8, shrink=0.8)
    
    # 2.4 预测结果（20×20，外推）
    ax9 = plt.subplot(3, 5, 9)
    pred_20x20 = predictions[20]
    im9 = ax9.imshow(pred_20x20, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100], origin='lower')
    ax9.set_title('⑨ Prediction (20×20)\nExtrapolation!', 
                  fontweight='bold', fontsize=11, color='darkorange')
    ax9.set_xlabel('x (mm)')
    ax9.set_ylabel('y (mm)')
    plt.colorbar(im9, ax=ax9, shrink=0.8)
    
    # 标记外推区域
    rect_outer = mpatches.Rectangle((0, 0), 100, 100, linewidth=2,
                                   edgecolor='red', facecolor='none',
                                   linestyle='--')
    ax9.add_patch(rect_outer)
    rect_inner = mpatches.Rectangle((25, 25), 50, 50, linewidth=2,
                                   edgecolor='cyan', facecolor='none')
    ax9.add_patch(rect_inner)
    
    # 2.5 对比：10×10 vs 目标
    ax10 = plt.subplot(3, 5, 10)
    # Resize 10×10预测到20×20
    pred_10x10_resized = zoom(pred_10x10, 2, order=1)
    ax10.plot(img_20x20_target.flatten(), 'b-', linewidth=1.5, alpha=0.7, label='Target 20×20')
    ax10.plot(pred_10x10_resized.flatten(), 'g--', linewidth=1.5, alpha=0.7, label='Pred 10×10 (resized)')
    ax10.plot(pred_20x20.flatten(), 'r:', linewidth=1.5, alpha=0.7, label='Pred 20×20')
    ax10.set_title('⑩ Flatten Comparison', fontweight='bold', fontsize=11)
    ax10.set_xlabel('Spatial Index')
    ax10.set_ylabel('Probability')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # ========== 第三行：误差分析 ==========
    
    # 3.1 10×10误差图（训练分辨率）
    ax11 = plt.subplot(3, 5, 11)
    # 提取目标的中心10×10
    target_center = img_20x20_target[5:15, 5:15]
    error_10x10 = np.abs(pred_10x10 - target_center)
    im11 = ax11.imshow(error_10x10, cmap='hot', vmin=0, vmax=0.5,
                       extent=[0, 100, 0, 100], origin='lower')
    mae_10 = error_10x10.mean()
    ax11.set_title(f'⑪ Error (10×10)\nMAE={mae_10:.4f}', 
                   fontweight='bold', fontsize=11)
    ax11.set_xlabel('x (mm)')
    ax11.set_ylabel('y (mm)')
    plt.colorbar(im11, ax=ax11, shrink=0.8)
    
    # 3.2 20×20误差图（完全外推）
    ax12 = plt.subplot(3, 5, 12)
    error_20x20 = np.abs(pred_20x20 - img_20x20_target)
    im12 = ax12.imshow(error_20x20, cmap='hot', vmin=0, vmax=0.5,
                       extent=[0, 100, 0, 100], origin='lower')
    mae_20 = error_20x20.mean()
    ax12.set_title(f'⑫ Error (20×20)\nMAE={mae_20:.4f}', 
                   fontweight='bold', fontsize=11)
    ax12.set_xlabel('x (mm)')
    ax12.set_ylabel('y (mm)')
    plt.colorbar(im12, ax=ax12, shrink=0.8)
    
    # 3.3 区域误差统计
    ax13 = plt.subplot(3, 5, 13)
    # 计算中心和边缘误差
    center_mask = np.zeros_like(error_20x20, dtype=bool)
    center_mask[5:15, 5:15] = True
    
    error_center = error_20x20[center_mask].mean()
    error_edge = error_20x20[~center_mask].mean()
    
    regions = ['Center\n(10×10)', 'Edge\n(10×10 border)']
    errors = [error_center, error_edge]
    colors = ['green', 'orange']
    
    bars = ax13.bar(regions, errors, color=colors, alpha=0.7, edgecolor='black')
    ax13.set_ylabel('MAE', fontsize=10)
    ax13.set_title('⑬ Regional Error\nCenter vs Edge', fontweight='bold', fontsize=11)
    ax13.set_ylim(0, max(errors) * 1.3)
    
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax13.text(bar.get_x() + bar.get_width()/2., height,
                 f'{err:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax13.grid(axis='y', alpha=0.3)
    
    # 3.4 统计信息
    ax14 = plt.subplot(3, 5, 14)
    ax14.axis('off')
    
    stats_text = f"""
    📊 Performance Metrics
    
    ━━━━━━━━━━━━━━━━━━━━━━━
    
    Training Setup:
    • Input: 5×5×100 sensors
    • Supervision: 10×10 center
    • Unseen: 20×20 edges
    
    Prediction@10×10:
    • MAE: {mae_10:.5f}
    • Status: ✓ Trained
    
    Prediction@20×20:
    • MAE: {mae_20:.5f}
    • Center MAE: {error_center:.5f}
    • Edge MAE: {error_edge:.5f}
    • Status: ⚠ Extrapolated
    
    ━━━━━━━━━━━━━━━━━━━━━━━
    
    💡 Key Observation:
    • Center: High accuracy
    • Edge: Degraded but still
      reasonable prediction
    • DeepONet learns continuous
      operator, not discrete map!
    """
    
    ax14.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax14.transAxes)
    
    # 3.5 损伤分布对比
    ax15 = plt.subplot(3, 5, 15)
    
    # 绘制径向分布
    center_y, center_x = 10, 10  # 中心坐标（20×20网格）
    y_coords, x_coords = np.ogrid[:20, :20]
    radius = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # 按半径分组
    radii = np.linspace(0, 14, 8)
    target_radial = []
    pred_radial = []
    
    for i in range(len(radii)-1):
        mask_ring = (radius >= radii[i]) & (radius < radii[i+1])
        target_radial.append(img_20x20_target[mask_ring].mean())
        pred_radial.append(pred_20x20[mask_ring].mean())
    
    r_centers = (radii[:-1] + radii[1:]) / 2
    
    ax15.plot(r_centers, target_radial, 'o-', linewidth=2, label='Target', color='blue')
    ax15.plot(r_centers, pred_radial, 's--', linewidth=2, label='Prediction', color='red')
    ax15.axvline(7, color='cyan', linestyle='--', linewidth=1.5, label='Supervised boundary')
    ax15.set_xlabel('Radius from center', fontsize=10)
    ax15.set_ylabel('Mean probability', fontsize=10)
    ax15.set_title('⑮ Radial Distribution\nCenter→Edge', fontweight='bold', fontsize=11)
    ax15.legend(fontsize=8)
    ax15.grid(True, alpha=0.3)
    
    # ========== 总标题 ==========
    mode_str = "Full 10×10 Input" if no_crop_input else "Cropped 5×5 Input"
    plt.suptitle(
        f'DeepONet Subgrid Training & Spatial Extrapolation Flow\n'
        f'Train: {mode_str} → 10×10 Center | Test: Query 20×20 Full | Sample {sample_idx}',
        fontsize=15, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Subgrid training flow visualization saved to {save_path}")
    plt.close()
    
    # 打印详细指标
    print("\n" + "="*70)
    print("Detailed Performance Metrics")
    print("="*70)
    print(f"Training Resolution (10×10):")
    print(f"  - MAE: {mae_10:.6f}")
    print(f"  - RMSE: {np.sqrt((error_10x10**2).mean()):.6f}")
    print(f"\nExtrapolation (20×20):")
    print(f"  - Overall MAE: {mae_20:.6f}")
    print(f"  - Center MAE: {error_center:.6f}")
    print(f"  - Edge MAE: {error_edge:.6f}")
    print(f"  - Edge/Center Ratio: {error_edge/error_center:.2f}x")
    print("="*70)
