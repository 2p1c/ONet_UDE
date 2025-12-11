"""
可视化对比函数模块

生成DeepONet vs CNN的对比图表
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_scenario_comparison(results, save_dir):
    """
    可视化单个场景的对比结果
    
    生成4子图:
    1. Ground Truth
    2. DeepONet Prediction
    3. CNN Prediction
    4. Error Difference
    
    Args:
        results: 对比结果字典
        save_dir: 保存目录
    """
    scenario_name = results['scenario']
    
    # 选择第一个样本可视化
    idx = 0
    img_true = results['true_images'][idx]
    pred_deeponet = results['deeponet']['preds'][idx]
    pred_cnn = results['cnn']['preds'][idx]
    
    mae_deeponet = results['deeponet']['maes'][idx]
    mae_cnn = results['cnn']['maes'][idx]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Ground Truth
    im0 = axes[0,0].imshow(img_true, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0,0].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[0,0].axis('off')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    # 2. DeepONet
    im1 = axes[0,1].imshow(pred_deeponet, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0,1].set_title(f'DeepONet Prediction\nMAE={mae_deeponet:.4f} (Direct)', 
                        fontsize=13, fontweight='bold')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # 3. CNN
    im2 = axes[1,0].imshow(pred_cnn, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[1,0].set_title(f'CNN Prediction\nMAE={mae_cnn:.4f} (w/ Interpolation)', 
                        fontsize=13, fontweight='bold')
    axes[1,0].axis('off')
    plt.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    # 4. Error Difference
    error_deeponet = np.abs(pred_deeponet - img_true)
    error_cnn = np.abs(pred_cnn - img_true)
    error_diff = error_cnn - error_deeponet
    
    im3 = axes[1,1].imshow(error_diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2, origin='lower')
    axes[1,1].set_title(f'Error Difference\n(Red=CNN worse, Blue=CNN better)', 
                        fontsize=13, fontweight='bold')
    axes[1,1].axis('off')
    cbar = plt.colorbar(im3, ax=axes[1,1], fraction=0.046, pad=0.04)
    cbar.set_label('CNN Error - DeepONet Error', fontsize=10)
    
    # 总标题
    plt.suptitle(f'Scenario: {scenario_name}', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{scenario_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_summary_table(all_results, save_dir):
    """
    生成对比汇总表格
    
    Args:
        all_results: 所有场景的结果列表
        save_dir: 保存目录
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # 表格数据
    table_data = []
    headers = ['Scenario', 'Config', 'DeepONet MAE', 'CNN MAE', 'Accuracy Loss', 
               'DeepONet Time', 'CNN Time', 'Time Overhead']
    
    for res in all_results:
        scenario = res['scenario']
        config = res.get('config_str', 'N/A')
        
        mae_onet = np.mean(res['deeponet']['maes'])
        mae_cnn = np.mean(res['cnn']['maes'])
        acc_loss = (mae_cnn / mae_onet - 1) * 100
        
        time_onet = np.mean(res['deeponet']['times']) * 1000  # ms
        time_cnn = np.mean(res['cnn']['times']) * 1000
        time_overhead = (time_cnn / time_onet - 1) * 100 if time_onet > 0 else 0
        
        row = [
            scenario,
            config,
            f'{mae_onet:.5f}',
            f'{mae_cnn:.5f}',
            f'+{acc_loss:.1f}%' if acc_loss > 0 else f'{acc_loss:.1f}%',
            f'{time_onet:.1f}ms',
            f'{time_cnn:.1f}ms',
            f'+{time_overhead:.1f}%' if time_overhead > 0 else f'{time_overhead:.1f}%'
        ]
        table_data.append(row)
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.suptitle('DeepONet vs CNN Generalization Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'summary_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
