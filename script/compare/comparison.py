"""
DeepONet vs CNN 泛化性对比脚本 - MVP版本

对比3个关键场景:
1. 基线 (5×5×100)
2. 稀疏传感器 (3×3×100)
3. 高分辨率输出 (20×20)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import time
import json

from data.dataset_simple import SimpleUSDataset3D
from nn.deeponet import DeepONet
from nn.cnn import SimpleCNN
from script.compare.interpolation_utils import interpolate_spatial, interpolate_output_image
from script.compare.visualization_comparison import visualize_scenario_comparison, generate_summary_table


def predict_deeponet(model, sig, output_size, device, train_config=None):
    """
    DeepONet预测 - 直接查询任意分辨率
    
    【修复】如果sig维度与训练不匹配,需要插值branch输入
    """
    model.eval()
    
    # 【新增】检查branch输入维度,必要时插值
    if train_config is not None:
        expected_shape = (train_config['ny'], train_config['nx'], train_config['sig_len'])
        if sig.shape != expected_shape:
            # 空间插值
            if sig.shape[:2] != (train_config['ny'], train_config['nx']):
                sig = interpolate_spatial(sig, (train_config['ny'], train_config['nx']))
            # 时间插值(如果需要)
            if sig.shape[2] != train_config['sig_len']:
                from script.compare.interpolation_utils import interpolate_temporal
                sig = interpolate_temporal(sig, train_config['sig_len'])
    
    branch_vec = sig.flatten()
    branch_batch = torch.from_numpy(branch_vec).unsqueeze(0).to(device)
    
    img_pred = np.zeros((output_size, output_size), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(output_size):
            for j in range(output_size):
                x_norm = j / (output_size - 1) if output_size > 1 else 0.0
                y_norm = i / (output_size - 1) if output_size > 1 else 0.0
                trunk_vec = torch.tensor([[x_norm, y_norm]], dtype=torch.float32).to(device)
                
                x_input = torch.cat([branch_batch, trunk_vec], dim=1)
                pred_val = model(x_input).cpu().numpy()[0, 0]
                img_pred[i, j] = pred_val
    
    return img_pred


def predict_cnn(model, sig, train_config, output_size, device):
    """
    CNN预测 - 需要插值到训练维度
    """
    model.eval()
    
    interp_start = time.time()
    
    # 1. 输入插值
    if sig.shape[:2] != (train_config['ny'], train_config['nx']):
        sig_interp = interpolate_spatial(sig, (train_config['ny'], train_config['nx']))
    else:
        sig_interp = sig
    
    # 2. 转换格式
    sig_cnn = np.transpose(sig_interp, (2, 0, 1))
    sig_tensor = torch.from_numpy(sig_cnn).unsqueeze(0).to(device)
    
    interp_time_input = time.time() - interp_start
    
    # 3. CNN预测
    with torch.no_grad():
        pred = model(sig_tensor)
        pred_np = pred.squeeze().cpu().numpy()
    
    # 4. 输出插值
    interp_start = time.time()
    if pred_np.shape[0] != output_size:
        pred_final = interpolate_output_image(pred_np, output_size)
    else:
        pred_final = pred_np
    
    interp_time_output = time.time() - interp_start
    total_interp_time = interp_time_input + interp_time_output
    
    return pred_final, total_interp_time


def compare_scenario(deeponet_model, cnn_model, test_dataset, train_config, 
                     scenario_name, output_size, device):
    """
    在单个场景下对比DeepONet和CNN
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    results = {
        'scenario': scenario_name,
        'deeponet': {'preds': [], 'maes': [], 'times': [], 'interp_times': []},  # 【修改】添加interp_times
        'cnn': {'preds': [], 'maes': [], 'times': [], 'interp_times': []},
        'true_images': []
    }
    
    n_test = min(5, len(test_dataset))
    
    for i in range(n_test):
        sig, img_true = test_dataset[i]
        
        # DeepONet预测 【修改】添加插值时间统计
        start = time.time()
        interp_start = time.time()
        
        # 检查是否需要插值
        expected_shape = (train_config['ny'], train_config['nx'], train_config['sig_len'])
        need_interp = sig.shape != expected_shape
        
        pred_deeponet = predict_deeponet(deeponet_model, sig, output_size, device, train_config)
        
        interp_time_deeponet = time.time() - interp_start if need_interp else 0.0
        time_deeponet = time.time() - start
        
        # 调整真值尺寸
        if img_true.shape[0] != output_size:
            img_true_resized = interpolate_output_image(img_true, output_size)
        else:
            img_true_resized = img_true
        
        mae_deeponet = np.mean(np.abs(pred_deeponet - img_true_resized))
        
        # CNN预测
        start = time.time()
        pred_cnn, interp_time = predict_cnn(cnn_model, sig, train_config, output_size, device)
        time_cnn = time.time() - start
        
        mae_cnn = np.mean(np.abs(pred_cnn - img_true_resized))
        
        # 保存结果
        results['deeponet']['preds'].append(pred_deeponet)
        results['deeponet']['maes'].append(mae_deeponet)
        results['deeponet']['times'].append(time_deeponet)
        results['deeponet']['interp_times'].append(interp_time_deeponet)  # 【新增】
        
        results['cnn']['preds'].append(pred_cnn)
        results['cnn']['maes'].append(mae_cnn)
        results['cnn']['times'].append(time_cnn)
        results['cnn']['interp_times'].append(interp_time)
        
        results['true_images'].append(img_true_resized)
    
    # 统计
    avg_mae_deeponet = np.mean(results['deeponet']['maes'])
    avg_mae_cnn = np.mean(results['cnn']['maes'])
    avg_time_deeponet = np.mean(results['deeponet']['times'])
    avg_time_cnn = np.mean(results['cnn']['times'])
    avg_interp_time_deeponet = np.mean(results['deeponet']['interp_times'])  # 【新增】
    avg_interp_time_cnn = np.mean(results['cnn']['interp_times'])
    
    # 【修改】打印信息,显示DeepONet的插值时间
    if avg_interp_time_deeponet > 0:
        print(f"\n{'DeepONet':12s} | MAE: {avg_mae_deeponet:.6f} | Time: {avg_time_deeponet*1000:.2f}ms (Interp: {avg_interp_time_deeponet*1000:.2f}ms)")
    else:
        print(f"\n{'DeepONet':12s} | MAE: {avg_mae_deeponet:.6f} | Time: {avg_time_deeponet*1000:.2f}ms (No Interp)")
    
    print(f"{'CNN':12s} | MAE: {avg_mae_cnn:.6f} | Time: {avg_time_cnn*1000:.2f}ms (Interp: {avg_interp_time_cnn*1000:.2f}ms)")
    print(f"{'Accuracy Gap':12s} | CNN worse by: {(avg_mae_cnn/avg_mae_deeponet - 1)*100:.1f}%")
    print(f"{'Time Overhead':12s} | CNN slower by: {(avg_time_cnn/avg_time_deeponet - 1)*100:.1f}%")
    
    return results


def main():
    print("="*70)
    print("DeepONet vs CNN Generalization Comparison - MVP")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # 加载模型
    train_config = {'nx': 5, 'ny': 5, 'sig_len': 100}
    
    deeponet = DeepONet(branch_dim=5*5*100, trunk_dim=2, branch_depth=2, 
                        trunk_depth=3, width=100, dropout=0.15).to(device)
    deeponet.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    deeponet.eval()
    print("✓ DeepONet loaded")
    
    cnn = SimpleCNN(input_channels=100, hidden_channels=64, dropout=0.15).to(device)
    cnn.load_state_dict(torch.load('checkpoints/best_cnn_model.pth', map_location=device))
    cnn.eval()
    print("✓ CNN loaded")
    
    # 场景1: 基线
    dataset_baseline = SimpleUSDataset3D(n_samples=10, nx=5, ny=5, sig_len=100, img_size=10, precompute=True)
    results_baseline = compare_scenario(deeponet, cnn, dataset_baseline, train_config,
                                       'Baseline (5x5x100)', 10, device)
    results_baseline['config_str'] = '5×5×100 → 10×10'
    
    # 场景2: 稀疏传感器
    dataset_sparse = SimpleUSDataset3D(n_samples=10, nx=3, ny=3, sig_len=100, img_size=10, precompute=True)
    results_sparse = compare_scenario(deeponet, cnn, dataset_sparse, train_config,
                                     'Sparse Sensors (3x3x100)', 10, device)
    results_sparse['config_str'] = '3×3×100 → 10×10'
    
    # 场景3: 高分辨率输出
    dataset_highres = SimpleUSDataset3D(n_samples=10, nx=5, ny=5, sig_len=100, img_size=50, precompute=True)
    results_highres = compare_scenario(deeponet, cnn, dataset_highres, train_config,
                                      'High-Resolution Output (50x50)', 50, device)
    results_highres['config_str'] = '5×5×100 → 50×50'
    
    # 可视化
    save_dir = 'images/compare_results'
    visualize_scenario_comparison(results_baseline, save_dir)
    visualize_scenario_comparison(results_sparse, save_dir)
    visualize_scenario_comparison(results_highres, save_dir)
    
    all_results = [results_baseline, results_sparse, results_highres]
    generate_summary_table(all_results, save_dir)
    
    # 保存JSON
    summary = {
        'baseline': {'deeponet_mae': float(np.mean(results_baseline['deeponet']['maes'])),
                    'cnn_mae': float(np.mean(results_baseline['cnn']['maes']))},
        'sparse': {'deeponet_mae': float(np.mean(results_sparse['deeponet']['maes'])),
                  'cnn_mae': float(np.mean(results_sparse['cnn']['maes']))},
        'highres': {'deeponet_mae': float(np.mean(results_highres['deeponet']['maes'])),
                   'cnn_mae': float(np.mean(results_highres['cnn']['maes']))}
    }
    with open(os.path.join(save_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Comparison completed!")
    print(f"📁 Results: {os.path.abspath(save_dir)}")
    print("="*70)


if __name__ == "__main__":
    main()
