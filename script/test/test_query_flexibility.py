"""
测试DeepONet的查询灵活性

对比实验：
1. CNN: 只能输出5×5
2. DeepONet: 可以输出5×5, 10×10, 20×20等任意分辨率
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import json

from data.dataset_simple import SimpleUSDataset3D
from data.transform import create_subgrid_dataset
from nn.deeponet import DeepONet
from nn.cnn import SimpleCNN


def test_deeponet_query(
    model,
    dataset,
    cropper,
    sample_idx: int,
    device,
    query_sizes: list = [5, 10, 20]
):
    """
    测试DeepONet在不同查询分辨率下的预测
    
    Args:
        model: 训练好的DeepONet
        dataset: 原始10×10数据集
        cropper: 子网格裁剪器
        sample_idx: 测试样本索引
        device: 设备
        query_sizes: 查询分辨率列表
    
    Returns:
        predictions: {size: pred_array}
    """
    model.eval()
    
    # 获取原始数据（10×10传感器网格）
    sig_full, img_target = dataset[sample_idx]
    
    # 裁剪到5×5用于Branch输入
    sig_5x5, _ = cropper.crop_signal(sig_full, return_grid=False)
    sig_input = sig_5x5.flatten()
    
    predictions = {}
    
    with torch.no_grad():
        for size in query_sizes:
            print(f"  → Querying {size}×{size} grid...")
            
            # 构建查询网格
            x_grid = np.linspace(0, 1, size)
            y_grid = np.linspace(0, 1, size)
            xv, yv = np.meshgrid(x_grid, y_grid, indexing='xy')
            
            pred_img = np.zeros((size, size))
            
            # 逐点查询
            for i in range(size):
                for j in range(size):
                    trunk_input = np.array([xv[i, j], yv[i, j]], dtype=np.float32)
                    x_input = np.concatenate([sig_input, trunk_input])
                    x_tensor = torch.from_numpy(x_input).unsqueeze(0).to(device)
                    
                    pred_val = model(x_tensor).cpu().numpy()[0, 0]
                    pred_img[i, j] = pred_val
            
            pred_img = np.clip(pred_img, 0, 1)
            predictions[size] = pred_img
    
    return predictions, img_target


def test_cnn_limitation(
    model,
    dataset,
    cropper,
    sample_idx: int,
    device
):
    """
    演示CNN的输出尺寸限制
    
    CNN只能输出训练时的固定尺寸（5×5或10×10）
    """
    model.eval()
    
    # 获取数据
    sig_full, img_target = dataset[sample_idx]
    sig_5x5, _ = cropper.crop_signal(sig_full, return_grid=True)
    
    with torch.no_grad():
        # CNN输入: (1, 100, 5, 5)
        sig_tensor = torch.FloatTensor(sig_5x5).permute(2, 0, 1).unsqueeze(0).to(device)
        pred = model(sig_tensor).squeeze().cpu().numpy()
    
    return pred, img_target


def visualize_comparison(
    deeponet_preds: dict,
    cnn_pred: np.ndarray,
    img_target: np.ndarray,
    save_path: str = 'images/query_flexibility_comparison.png'
):
    """
    可视化对比：DeepONet多分辨率 vs CNN固定分辨率
    """
    fig = plt.figure(figsize=(20, 12))
    
    # ===== 第一行：DeepONet多分辨率查询 =====
    query_sizes = sorted(deeponet_preds.keys())
    n_sizes = len(query_sizes)
    
    for idx, size in enumerate(query_sizes):
        ax = plt.subplot(3, n_sizes, idx + 1)
        pred = deeponet_preds[size]
        
        im = ax.imshow(pred, cmap='hot', vmin=0, vmax=1, origin='lower',
                      extent=[0, 100, 0, 100])
        ax.set_title(f'DeepONet Query\n{size}×{size} Resolution', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 计算与目标的误差
        if size == 10:
            mae = np.abs(pred - img_target).mean()
            ax.text(0.5, -0.15, f'MAE={mae:.4f}', 
                   transform=ax.transAxes, ha='center', fontsize=10)
    
    # ===== 第二行：CNN固定输出 =====
    ax_cnn = plt.subplot(3, n_sizes, n_sizes + 2)
    im_cnn = ax_cnn.imshow(cnn_pred, cmap='hot', vmin=0, vmax=1, origin='lower',
                          extent=[0, 100, 0, 100])
    ax_cnn.set_title(f'CNN Output\n{cnn_pred.shape[0]}×{cnn_pred.shape[1]} (Fixed)', 
                    fontweight='bold', fontsize=12, color='darkred')
    ax_cnn.set_xlabel('x (mm)')
    ax_cnn.set_ylabel('y (mm)')
    plt.colorbar(im_cnn, ax=ax_cnn, shrink=0.8)
    
    # 添加限制说明
    ax_cnn.text(0.5, -0.15, '❌ Cannot change output size', 
               transform=ax_cnn.transAxes, ha='center', 
               fontsize=10, color='darkred', fontweight='bold')
    
    # ===== 第二行：目标真值 =====
    ax_target = plt.subplot(3, n_sizes, n_sizes + 3)
    im_target = ax_target.imshow(img_target, cmap='hot', vmin=0, vmax=1, origin='lower',
                                extent=[0, 100, 0, 100])
    ax_target.set_title('Ground Truth\n10×10', fontweight='bold', fontsize=12)
    ax_target.set_xlabel('x (mm)')
    ax_target.set_ylabel('y (mm)')
    plt.colorbar(im_target, ax=ax_target, shrink=0.8)
    
    # ===== 第三行：误差对比 =====
    for idx, size in enumerate(query_sizes):
        ax = plt.subplot(3, n_sizes, 2*n_sizes + idx + 1)
        pred = deeponet_preds[size]
        
        if size == 10:
            error = np.abs(pred - img_target)
            im_err = ax.imshow(error, cmap='hot', vmin=0, vmax=0.5, origin='lower',
                             extent=[0, 100, 0, 100])
            ax.set_title(f'Error Map ({size}×{size})', fontweight='bold', fontsize=11)
            plt.colorbar(im_err, ax=ax, shrink=0.8)
        else:
            # 插值到10×10再计算误差
            pred_interp = zoom(pred, 10/size, order=1)
            error = np.abs(pred_interp - img_target)
            im_err = ax.imshow(error, cmap='hot', vmin=0, vmax=0.5, origin='lower',
                             extent=[0, 100, 0, 100])
            ax.set_title(f'Error (Interp {size}→10)', fontweight='bold', fontsize=11)
            plt.colorbar(im_err, ax=ax, shrink=0.8)
    
    plt.suptitle('DeepONet Query Flexibility: Continuous Operator Learning\n'
                 'Training: 5×5 Input → 5×5 Output | Testing: 5×5 Input → Any Resolution',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved to {save_path}")
    plt.close()


def main():
    """主测试流程"""
    print("=" * 70)
    print("Testing Query Flexibility: DeepONet vs CNN")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 加载数据集 ====================
    print("\n📦 Loading 10×10 dataset...")
    dataset = SimpleUSDataset3D(
        n_samples=100,  # 测试集
        nx=10,
        ny=10,
        sig_len=100,
        img_size=10,
        precompute=True
    )
    
    # 创建5×5裁剪器
    _, cropper = create_subgrid_dataset(
        dataset,
        sub_nx=5,
        sub_ny=5,
        position='center',
        for_cnn=False,
        crop_target=False,  # 测试时不裁剪目标
        random_seed=42
    )
    
    # ==================== 加载训练好的模型 ====================
    print("\n🔧 Loading trained models...")
    
    # DeepONet
    config_path = 'checkpoints/last_config.json'
    with open(config_path, 'r') as f:
        deeponet_config = json.load(f)
    
    deeponet = DeepONet(
        branch_dim=deeponet_config['branch_dim'],
        trunk_dim=deeponet_config['trunk_dim'],
        branch_depth=deeponet_config['branch_depth'],
        trunk_depth=deeponet_config['trunk_depth'],
        width=deeponet_config['width'],
        activation='relu',
        initializer='Glorot normal',
        dropout=deeponet_config.get('dropout', 0.0)
    ).to(device)
    deeponet.load_state_dict(torch.load('checkpoints/best_model.pth'))
    print("✓ DeepONet loaded")
    
    # CNN
    cnn_config_path = 'checkpoints/last_cnn_config.json'
    with open(cnn_config_path, 'r') as f:
        cnn_config = json.load(f)
    
    cnn = SimpleCNN(
        input_channels=cnn_config['input_channels'],
        hidden_channels=cnn_config['hidden_channels'],
        dropout=cnn_config['dropout'],
        input_size=cnn_config['input_size']
    ).to(device)
    cnn.load_state_dict(torch.load('checkpoints/best_cnn_model.pth'))
    print("✓ CNN loaded")
    
    # ==================== 测试 ====================
    print("\n🧪 Testing query flexibility...")
    sample_idx = 0
    
    # DeepONet多分辨率查询
    print("\n🔹 DeepONet: Querying multiple resolutions...")
    deeponet_preds, img_target = test_deeponet_query(
        deeponet, dataset, cropper,
        sample_idx, device,
        query_sizes=[5, 10, 20, 50]
    )
    
    # CNN固定输出
    print("\n🔹 CNN: Fixed output size...")
    cnn_pred, _ = test_cnn_limitation(
        cnn, dataset, cropper,
        sample_idx, device
    )
    
    # ==================== 可视化对比 ====================
    print("\n📊 Generating comparison visualization...")
    visualize_comparison(
        deeponet_preds,
        cnn_pred,
        img_target,
        save_path='images/query_flexibility_comparison.png'
    )
    
    # ==================== 计算指标 ====================
    print("\n📈 Performance Metrics:")
    print("=" * 70)
    
    for size in sorted(deeponet_preds.keys()):
        pred = deeponet_preds[size]
        
        if size == 10:
            mae = np.abs(pred - img_target).mean()
            rmse = np.sqrt(np.mean((pred - img_target)**2))
            print(f"DeepONet {size}×{size}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    cnn_mae = np.abs(cnn_pred - img_target).mean()
    cnn_rmse = np.sqrt(np.mean((cnn_pred - img_target)**2))
    print(f"CNN {cnn_pred.shape[0]}×{cnn_pred.shape[1]}: MAE={cnn_mae:.6f}, RMSE={cnn_rmse:.6f}")
    
    print("\n" + "=" * 70)
    print("🎉 Query flexibility test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
