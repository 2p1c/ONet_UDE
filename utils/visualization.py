"""
可视化工具模块
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
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
    raw_dataset,
    sample_idx: int,
    device,
    save_path: str = 'images/train_prediction.png'
):
    """可视化预测结果（真实图 | 预测图 | 误差图）"""
    model.eval()
    
    # 获取原始样本
    sig, img_true = raw_dataset[sample_idx]
    
    # Branch输入
    branch_vec = sig.flatten()
    branch_batch = torch.from_numpy(branch_vec).unsqueeze(0).to(device)
    
    # 预测整个网格
    img_size = img_true.shape[0]
    img_pred = np.zeros((img_size, img_size), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(img_size):
            for j in range(img_size):
                x_norm = j / (img_size - 1)
                y_norm = i / (img_size - 1)
                trunk_vec = torch.tensor([[x_norm, y_norm]], dtype=torch.float32).to(device)
                
                x_input = torch.cat([branch_batch, trunk_vec], dim=1)
                pred_val = model(x_input).cpu().numpy()[0, 0]
                img_pred[i, j] = pred_val
    
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
    axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
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
    
    plt.suptitle(f'DeepONet Prediction (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Prediction saved to {save_path}")
    plt.close()
