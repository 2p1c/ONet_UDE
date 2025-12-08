"""
训练结果归档脚本

功能:
1. 读取最近的checkpoint
2. 保存训练配置和Loss信息
3. 归档可视化图像
4. 以时间戳命名存档文件夹
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
from datetime import datetime
from pathlib import Path
import torch


def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    """获取最新的checkpoint文件"""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' not found!")
    
    # 查找所有.pth文件
    ckpt_files = list(ckpt_path.glob('*.pth'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in '{checkpoint_dir}'!")
    
    # 按修改时间排序，获取最新的
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return latest_ckpt


def load_checkpoint_info(ckpt_path):
    """加载checkpoint信息"""
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # 尝试提取参数数量
        n_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        
        info = {
            'checkpoint_path': str(ckpt_path),
            'file_size_mb': ckpt_path.stat().st_size / (1024 * 1024),
            'modified_time': datetime.fromtimestamp(ckpt_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'n_parameters': n_params,
        }
        return info
    except Exception as e:
        print(f"Warning: Failed to load checkpoint details: {e}")
        return {'checkpoint_path': str(ckpt_path)}


def parse_train_config(checkpoint_dir='checkpoints'):
    """
    从保存的配置文件中读取训练配置
    
    Args:
        checkpoint_dir: checkpoint目录路径
    
    Returns:
        config字典，如果文件不存在则返回None
    """
    config_path = Path(checkpoint_dir) / 'last_config.json'
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("         Using default placeholder config")
        # 返回占位符配置
        return {
            'note': 'Config file not found, using placeholder',
            'n_samples': 'N/A',
            'train_ratio': 'N/A',
        }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Config loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        return {'error': str(e)}


def find_latest_images(image_dir='images'):
    """查找最新的可视化图像"""
    img_path = Path(image_dir)
    if not img_path.exists():
        print(f"Warning: Image directory '{image_dir}' not found!")
        return []
    
    # 查找所有图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.pdf']:
        image_files.extend(img_path.glob(ext))
    
    if not image_files:
        print(f"Warning: No image files found in '{image_dir}'!")
        return []
    
    # 按修改时间排序
    image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return image_files


def extract_best_loss(checkpoint_dir='checkpoints'):
    """
    从保存的指标文件中提取Loss信息
    
    Args:
        checkpoint_dir: checkpoint目录路径
    
    Returns:
        loss信息字典
    """
    metrics_path = Path(checkpoint_dir) / 'last_metrics.pth'
    
    if not metrics_path.exists():
        print(f"Warning: Metrics file not found at {metrics_path}")
        return {
            'best_train_loss': 'N/A',
            'best_test_loss': 'N/A',
            'final_train_loss': 'N/A',
            'final_test_loss': 'N/A',
            'n_epochs': 'N/A',
        }
    
    try:
        metrics = torch.load(metrics_path, map_location='cpu')
        
        loss_info = {
            'best_test_loss': f"{metrics.get('best_test_loss', 'N/A'):.6f}" if isinstance(metrics.get('best_test_loss'), float) else 'N/A',
            'final_train_loss': f"{metrics.get('final_train_loss', 'N/A'):.6f}" if isinstance(metrics.get('final_train_loss'), float) else 'N/A',
            'final_test_loss': f"{metrics.get('final_test_loss', 'N/A'):.6f}" if isinstance(metrics.get('final_test_loss'), float) else 'N/A',
            'n_epochs': metrics.get('n_epochs_completed', 'N/A'),
            'train_losses_length': len(metrics.get('train_losses', [])),
            'test_losses_length': len(metrics.get('test_losses', [])),
        }
        
        print(f"✓ Metrics loaded from: {metrics_path}")
        return loss_info
        
    except Exception as e:
        print(f"Warning: Failed to load metrics: {e}")
        return {'error': str(e)}


def create_archive(archive_dir='archives'):
    """创建归档文件夹"""
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = Path(archive_dir) / f'training_{timestamp}'
    archive_path.mkdir(parents=True, exist_ok=True)
    
    return archive_path, timestamp


def generate_report(archive_path, config, ckpt_info, loss_info, image_files):
    """生成Markdown格式的训练报告"""
    report_lines = []
    report_lines.append(f"# Training Results Report")
    report_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Checkpoint信息
    report_lines.append("## 1. Checkpoint Information\n")
    report_lines.append(f"- **File**: `{ckpt_info.get('checkpoint_path', 'N/A')}`")
    report_lines.append(f"- **Size**: {ckpt_info.get('file_size_mb', 0):.2f} MB")
    report_lines.append(f"- **Modified**: {ckpt_info.get('modified_time', 'N/A')}")
    report_lines.append(f"- **Parameters**: {ckpt_info.get('n_parameters', 'N/A'):,}\n")
    
    # 2. 训练配置
    report_lines.append("## 2. Training Configuration\n")
    report_lines.append("```json")
    report_lines.append(json.dumps(config, indent=2))
    report_lines.append("```\n")
    
    # 3. Loss信息
    report_lines.append("## 3. Training Metrics\n")
    report_lines.append(f"- **Epochs Completed**: {loss_info.get('n_epochs', 'N/A')}")
    report_lines.append(f"- **Best Test Loss**: {loss_info.get('best_test_loss', 'N/A')}")
    report_lines.append(f"- **Final Train Loss**: {loss_info.get('final_train_loss', 'N/A')}")
    report_lines.append(f"- **Final Test Loss**: {loss_info.get('final_test_loss', 'N/A')}\n")
    
    # 4. 可视化图像
    report_lines.append("## 4. Visualizations\n")
    if image_files:
        for img in image_files[:5]:  # 最多显示5张
            report_lines.append(f"- `{img.name}`")
    else:
        report_lines.append("- No visualization images found")
    
    report_lines.append("\n---\n*Generated by ONet_UDE Archive Script*")
    
    # 保存报告
    report_path = archive_path / 'README.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Report saved to: {report_path}")
    return report_path


def archive_results():
    """主归档流程"""
    print("=" * 70)
    print("Training Results Archive Script")
    print("=" * 70)
    
    try:
        # 1. 获取最新checkpoint
        print("\n[1/6] Finding latest checkpoint...")
        ckpt_path = get_latest_checkpoint()
        print(f"✓ Found: {ckpt_path}")
        
        # 2. 加载checkpoint信息
        print("\n[2/6] Loading checkpoint information...")
        ckpt_info = load_checkpoint_info(ckpt_path)
        print(f"✓ Checkpoint size: {ckpt_info.get('file_size_mb', 0):.2f} MB")
        
        # 3. 读取配置 - 【修改】从保存的文件读取
        print("\n[3/6] Loading training configuration...")
        config = parse_train_config()
        print(f"✓ Config loaded: {len(config)} parameters")
        
        # 4. 提取Loss信息 - 【修改】从保存的文件读取
        print("\n[4/6] Extracting loss information...")
        loss_info = extract_best_loss()
        print(f"✓ Best test loss: {loss_info.get('best_test_loss', 'N/A')}")
        
        # 5. 查找可视化图像
        print("\n[5/6] Finding visualization images...")
        image_files = find_latest_images()
        print(f"✓ Found {len(image_files)} image(s)")
        
        # 6. 创建归档
        print("\n[6/6] Creating archive...")
        archive_path, timestamp = create_archive()
        
        # 保存配置JSON
        config_path = archive_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved: {config_path}")
        
        # 保存checkpoint信息JSON
        ckpt_info_path = archive_path / 'checkpoint_info.json'
        with open(ckpt_info_path, 'w') as f:
            json.dump(ckpt_info, f, indent=2)
        print(f"✓ Checkpoint info saved: {ckpt_info_path}")
        
        # 保存Loss信息JSON
        loss_path = archive_path / 'loss_metrics.json'
        with open(loss_path, 'w') as f:
            json.dump(loss_info, f, indent=2)
        print(f"✓ Loss metrics saved: {loss_path}")
        
        # 复制checkpoint
        ckpt_archive_path = archive_path / ckpt_path.name
        shutil.copy2(ckpt_path, ckpt_archive_path)
        print(f"✓ Checkpoint copied: {ckpt_archive_path}")
        
        # 【新增】复制原始配置和指标文件
        src_config = Path('checkpoints') / 'last_config.json'
        src_metrics = Path('checkpoints') / 'last_metrics.pth'
        
        if src_config.exists():
            shutil.copy2(src_config, archive_path / 'last_config.json')
            print(f"✓ Original config copied")
        
        if src_metrics.exists():
            shutil.copy2(src_metrics, archive_path / 'last_metrics.pth')
            print(f"✓ Original metrics copied")
        
        # 复制可视化图像
        if image_files:
            img_archive_dir = archive_path / 'images'
            img_archive_dir.mkdir(exist_ok=True)
            for img in image_files[:10]:  # 最多复制10张最新的
                shutil.copy2(img, img_archive_dir / img.name)
            print(f"✓ Images copied: {len(image_files[:10])} files")
        
        # 生成报告
        generate_report(archive_path, config, ckpt_info, loss_info, image_files)
        
        # 完成
        print("\n" + "=" * 70)
        print(f"✅ Archive created successfully!")
        print(f"📁 Location: {archive_path.absolute()}")
        print("=" * 70)
        print("\nArchive contents:")
        print(f"  - config.json           (训练配置)")
        print(f"  - checkpoint_info.json  (模型信息)")
        print(f"  - loss_metrics.json     (Loss指标)")
        print(f"  - {ckpt_path.name}      (模型权重)")
        print(f"  - images/               (可视化图像)")
        print(f"  - README.md             (训练报告)")
        
        return archive_path
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    archive_results()
