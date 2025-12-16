"""
可视化工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from typing import List


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
    plt.title('DeepONet Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    可视化预测结果
    
    Args:
        model: 训练好的模型
        dataset: 原始数据集（完整信号）
        sample_idx: 样本索引
        device: 设备
        cropper: 裁剪器（可选），如果训练时用了裁剪，需要传入
            - SpatialCropper: 边界/随机裁剪
            - SquareCropper: 正方形裁剪
    """
    model.eval()
    
    # 获取原始数据
    sig_full, img_true = dataset[sample_idx]  # (ny, nx, sig_len), (img_size, img_size)
    
    # 【修复】根据cropper类型进行裁剪
    if cropper is not None:
        # 判断cropper类型
        from data.transform import SpatialCropper, SquareCropper
        
        if isinstance(cropper, SquareCropper):
            # SquareCropper: 返回flatten格式用于DeepONet
            sig_for_pred, kept_indices = cropper.crop_signal(
                sig_full, 
                return_grid=False  # DeepONet需要flatten
            )
            print(f"✓ Using SquareCropper: {sig_for_pred.shape}")
        elif isinstance(cropper, SpatialCropper):
            # SpatialCropper: 已经返回flatten格式
            sig_for_pred, kept_indices = cropper.crop_signal(
                sig_full, 
                random_per_sample=False
            )
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
                # 构建输入: [signal_flat, x, y]
                trunk_input = np.array([xv[i, j], yv[i, j]], dtype=np.float32)
                x_input = np.concatenate([sig_flat, trunk_input])
                x_input = torch.from_numpy(x_input).unsqueeze(0).to(device)
                
                # 预测
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
    ax3.set_title('Absolute Error', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 标题添加模式信息
    mode_str = "Cropped Input" if cropper is not None else "Full Input"
    plt.suptitle(f'DeepONet Prediction - Sample {sample_idx} ({mode_str})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存
    save_path = 'images/prediction_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Prediction visualization saved to {save_path}")
    plt.show()
    
    # 打印误差统计
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    print(f"\n✓ Prediction metrics:")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - Max error: {error.max():.6f}")
