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
    cropper=None
):
    """
    可视化DeepONet预测结果
    
    Args:
        model: 训练好的模型
        dataset: 原始数据集（完整信号）
        sample_idx: 样本索引
        device: 设备
        cropper: 裁剪器（可选）
    """
    model.eval()
    
    # 获取原始数据
    sig_full, img_true = dataset[sample_idx]
    
    # 根据cropper类型进行裁剪
    if cropper is not None:
        from data.transform import SpatialCropper, SquareCropper
        
        if isinstance(cropper, SquareCropper):
            sig_for_pred, kept_indices = cropper.crop_signal(sig_full, return_grid=False)
            print(f"✓ Using SquareCropper: {sig_for_pred.shape}")
        elif isinstance(cropper, SpatialCropper):
            sig_for_pred, kept_indices = cropper.crop_signal(sig_full, random_per_sample=False)
            print(f"✓ Using SpatialCropper: {sig_for_pred.shape}")
        else:
            raise TypeError(f"Unknown cropper type: {type(cropper)}")
        
        print(f"  Kept {len(kept_indices)} sensors out of {sig_full.shape[0] * sig_full.shape[1]}")
    else:
        sig_for_pred = sig_full
    
    # Flatten信号
    sig_flat = sig_for_pred.flatten()
    
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


def visualize_cnn_prediction(model, raw_dataset, sample_idx, device, 
                            save_path='images/cnn_prediction.png',
                            cropper=None):  # 【新增】cropper参数
    """
    CNN专用可视化函数
    
    Args:
        model: CNN模型
        raw_dataset: 原始数据集
        sample_idx: 样本索引
        device: 设备
        save_path: 保存路径
        cropper: 裁剪器（可选），如果训练时用了裁剪，需要传入
    """
    model.eval()
    
    # 获取原始样本
    sig_full, img_true = raw_dataset[sample_idx]
    
    # 【新增】如果有cropper，对信号进行裁剪
    if cropper is not None:
        from data.transform import SquareCropper
        
        if isinstance(cropper, SquareCropper):
            # 裁剪信号，保持网格格式
            sig, _ = cropper.crop_signal(sig_full, return_grid=True)
            print(f"✓ Using cropped signal for CNN prediction: {sig.shape}")
        else:
            raise TypeError(f"CNN only supports SquareCropper, got {type(cropper)}")
    else:
        sig = sig_full
        print(f"✓ Using full signal for CNN prediction: {sig.shape}")
    
    # 转换为CNN输入格式
    sig_cnn = np.transpose(sig, (2, 0, 1))  # (T, H, W)
    sig_tensor = torch.from_numpy(sig_cnn).unsqueeze(0).to(device)
    
    # 【新增】打印输入形状用于调试
    print(f"  CNN input tensor shape: {sig_tensor.shape}")
    
    # 预测
    with torch.no_grad():
        pred = model(sig_tensor)
        img_pred = pred.squeeze().cpu().numpy()
    
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
    
    # 【修改】标题说明输入模式
    mode_str = f"Cropped ({sig.shape[0]}×{sig.shape[1]})" if cropper is not None else f"Full ({sig.shape[0]}×{sig.shape[1]})"
    plt.suptitle(f'CNN Prediction - Sample {sample_idx} ({mode_str})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Prediction visualization saved to {save_path}")
    plt.close()
    
    # 【新增】打印预测统计信息
    print(f"\nPrediction statistics:")
    print(f"  - Min: {img_pred.min():.6f}")
    print(f"  - Max: {img_pred.max():.6f}")
    print(f"  - Mean: {img_pred.mean():.6f}")
    print(f"  - Std: {img_pred.std():.6f}")
    print(f"  - MAE: {error.mean():.6f}")
    print(f"  - RMSE: {np.sqrt(np.mean(error**2)):.6f}")


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
       2500 → 900 (36% retained)
    
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
    
    # RMS能量分布
    ax8 = plt.subplot(2, 4, 8)
    rms_full = np.sqrt(np.mean(sig_full**2, axis=2))
    rms_full_interp = zoom(rms_full, 4, order=1)
    im8 = ax8.imshow(rms_full_interp, cmap='viridis',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    
    if cropper.crop_position == 'center':
        rect_x = [20, 80, 80, 20, 20]
        rect_y = [20, 20, 80, 80, 20]
        ax8.plot(rect_x, rect_y, 'r-', linewidth=3, label='Kept region')
    
    ax8.set_title('⑧ RMS Energy Distribution\n(Full grid)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('x (mm)')
    ax8.set_ylabel('y (mm)')
    plt.colorbar(im8, ax=ax8, shrink=0.8, label='RMS Energy')
    ax8.legend(fontsize=9)
    
    plt.suptitle(f'DeepONet Cropped Dataset Visualization (Sample {sample_idx})\n'
                 f'Crop Mode: {cropper.crop_position} | Kept: 9/25 sensors (36%)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()


def visualize_cropped_dataset_cnn(raw_dataset, cropper, sample_idx=0, save_path='images/dataset_check/cnn_cropped_data.png'):
    """可视化CNN使用的裁剪数据集"""
    print(f"\n📊 Visualizing cropped dataset for CNN...")
    
    sig_full, img_target = raw_dataset[sample_idx]
    sig_cropped, kept_indices = cropper.crop_signal(sig_full, return_grid=True)
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
    
    text_info = f"""
    📊 CNN Input Transformation
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Original Signal:
       • Shape: (5, 5, 100)
       • Total: 2500 time samples
       • Input channels: 5
    
    ✂️ After Cropping:
       • Shape: (3, 3, 100) → flatten
       • Total: 900 time samples
       • Input channels: 3
    
    📉 Dimension Reduction:
       2500 → 900 (36% retained)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🎯 Target (Unchanged):
       • Shape: (10, 10)
       • Total: 100 output points
    
    💡 Key Insight:
       CNN直接学习局部特征，
       无需推断全局信息。
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
    
    if cropper.crop_position == 'center':
        rect_x = [20, 80, 80, 20, 20]
        rect_y = [20, 20, 80, 80, 20]
        ax8.plot(rect_x, rect_y, 'r-', linewidth=3, label='Kept region')
    
    ax8.set_title('⑧ RMS Energy Distribution\n(Full grid)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('x (mm)')
    ax8.set_ylabel('y (mm)')
    plt.colorbar(im8, ax=ax8, shrink=0.8, label='RMS Energy')
    ax8.legend(fontsize=9)
    
    plt.suptitle(f'CNN Cropped Dataset Visualization (Sample {sample_idx})\n'
                 f'Crop Mode: {cropper.crop_position} | Kept: 9/25 sensors (36%)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
