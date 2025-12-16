"""
Dataset Module Tests - 3D Dataset Only

Test Coverage:
- ToyUSDataset3D basic functionality (3D spatiotemporal)
- Spectrogram transforms (3D)
- Visualization tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure matplotlib for better compatibility
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
plt.rcParams['figure.max_open_warning'] = 50

# Use correct import path
from data.dataset import ToyUSDataset3D, batch_spectrogram_3d
from data.dataset_simple import SimpleUSDataset3D
from data.transform import create_cropped_dataset, SpatialCropper  # 新增导入


def test_dataset_3d_basic():
    """Test 3D dataset basic functionality"""
    print("=" * 60)
    print("Test 1: 3D Dataset Basic Functionality")
    print("=" * 60)
    
    dataset = ToyUSDataset3D(
        n_samples=5,
        nx=21,
        ny=21,
        sig_len=500,
        img_size=32,
        precompute=True
    )
    
    # Check length
    assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"
    print(f"✓ Dataset length correct: {len(dataset)}")
    
    # Check single sample
    sig_3d, img = dataset[0]
    assert sig_3d.shape == (21, 21, 500), f"Signal shape error: {sig_3d.shape}"
    assert img.shape == (32, 32), f"Image shape error: {img.shape}"
    print(f"✓ Sample shapes correct: signal{sig_3d.shape}, image{img.shape}")
    
    # Check value ranges
    assert sig_3d.min() >= -1 and sig_3d.max() <= 1, "Signal values out of range"
    assert img.min() >= 0 and img.max() <= 1, "Image values out of range"
    print(f"✓ Value ranges correct")
    
    # Test info
    info = dataset.get_info()
    print(f"✓ Dataset info: {info}")
    
    print("✅ 3D dataset test passed!\n")


def test_spectrogram_3d():
    """Test 3D spectrogram transform"""
    print("=" * 60)
    print("Test 2: 3D Spectrogram Transform")
    print("=" * 60)
    
    # Create test data
    dataset = ToyUSDataset3D(n_samples=2, nx=11, ny=11, sig_len=500)
    sig_3d, _ = dataset[0]
    sig_3d_batch = np.stack([dataset[i][0] for i in range(2)])
    
    # Test single sample
    spec_3d = batch_spectrogram_3d(sig_3d, n_fft=128, hop_length=4)
    print(f"✓ Single sample spectrogram shape: {spec_3d.shape}")
    assert spec_3d.shape[0] == 1, "Batch dimension error"
    assert spec_3d.shape[1] == 11 and spec_3d.shape[2] == 11, "Spatial dimension error"
    
    # Test batch
    spec_3d_batch = batch_spectrogram_3d(sig_3d_batch, n_fft=128, hop_length=4)
    print(f"✓ Batch spectrogram shape: {spec_3d_batch.shape}")
    assert spec_3d_batch.shape[0] == 2, "Batch size error"
    
    # Check values
    assert not torch.isnan(spec_3d_batch).any(), "Spectrogram contains NaN"
    assert not torch.isinf(spec_3d_batch).any(), "Spectrogram contains Inf"
    print(f"✓ Spectrogram values normal")
    
    print("✅ 3D spectrogram transform test passed!\n")


def visualize_3d_dataset():
    """Visualize 3D dataset - a) time signal, b) spatial distribution, c) spectrum, d) defect label"""
    print("=" * 60)
    print("Visualization: 3D Dataset Complete Analysis")
    print("=" * 60)
    
    # Create dataset
    dataset = ToyUSDataset3D(
        n_samples=1,
        nx=51,
        ny=51,
        sig_len=800,
        img_size=64,
        max_defects=3,
        precompute=True
    )
    
    sig_3d, img = dataset[0]
    print(f"Data shapes: signal{sig_3d.shape}, image{img.shape}")
    
    # Compute spectrogram
    spec_3d = batch_spectrogram_3d(sig_3d, n_fft=256, hop_length=8)
    spec_3d = spec_3d.squeeze(0).numpy()  # (ny, nx, n_freq, n_frames)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # ===== a) Time domain signal at a point =====
    ax1 = plt.subplot(3, 3, 1)
    sample_y, sample_x = 25, 25  # Center point
    time_signal = sig_3d[sample_y, sample_x, :]
    t_vec = np.linspace(0, dataset.T, dataset.sig_len)
    ax1.plot(t_vec * 1000, time_signal, linewidth=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'a) Time Signal at Point ({sample_x},{sample_y})')
    ax1.grid(True, alpha=0.3)
    
    # ===== b) Spatial distribution at a time instant =====
    ax2 = plt.subplot(3, 3, 2)
    time_idx = 150
    spatial_snapshot = sig_3d[:, :, time_idx]
    
    # 对空间分布图进行4倍线性插值
    from scipy.ndimage import zoom
    spatial_snapshot_interp = zoom(spatial_snapshot, 4, order=1)  # order=1 表示线性插值
    
    # 绘制插值后的图像
    im2 = ax2.imshow(spatial_snapshot_interp, cmap='seismic', 
                     extent=[0, dataset.Lx, 0, dataset.Ly],
                     origin='lower', aspect='equal')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title(f'b) Spatial Distribution at t={t_vec[time_idx]*1000:.2f}ms (4x Interpolated)')
    plt.colorbar(im2, ax=ax2, label='Amplitude')
    
    # ===== b2) 3D spatial distribution =====
    ax3 = plt.subplot(3, 3, 3, projection='3d')
    x_coords = np.linspace(0, dataset.Lx, dataset.nx)
    y_coords = np.linspace(0, dataset.Ly, dataset.ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    surf = ax3.plot_surface(X, Y, spatial_snapshot, cmap='seismic', alpha=0.8)
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('Amplitude')
    ax3.set_title('b2) 3D Spatial Distribution')
    
    # ===== c) Spectral features =====
    # c1: Single point spectrum
    ax4 = plt.subplot(3, 3, 4)
    point_spec = spec_3d[sample_y, sample_x, :, :]  # (n_freq, n_frames)
    im4 = ax4.imshow(point_spec, aspect='auto', origin='lower', cmap='viridis')
    ax4.set_xlabel('Time Frame')
    ax4.set_ylabel('Frequency Bin')
    ax4.set_title(f'c1) STFT Spectrum at Point ({sample_x},{sample_y})')
    plt.colorbar(im4, ax=ax4, label='Magnitude (log)')
    
    # c2: Spectral energy spatial distribution
    ax5 = plt.subplot(3, 3, 5)
    spec_energy = np.mean(spec_3d, axis=(2, 3))  # Average spectral energy
    
    # 对频谱能量分布图进行4倍线性插值
    spec_energy_interp = zoom(spec_energy, 4, order=1)  # order=1 表示线性插值
    
    # 绘制插值后的图像
    im5 = ax5.imshow(spec_energy_interp, cmap='hot', 
                     extent=[0, dataset.Lx, 0, dataset.Ly],
                     origin='lower', aspect='equal')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('y (m)')
    ax5.set_title('c2) Average Spectral Energy Distribution (4x Interpolated)')
    plt.colorbar(im5, ax=ax5, label='Energy')
    
    # ===== 【新增】c3: RMS Energy Map (时域RMS) =====
    ax6 = plt.subplot(3, 3, 6)
    # 计算每个空间点的时域RMS能量
    rms_energy = np.sqrt(np.mean(sig_3d**2, axis=2))  # (ny, nx)
    rms_energy_interp = zoom(rms_energy, 4, order=1)  # 4x插值
    
    im6 = ax6.imshow(rms_energy_interp, cmap='hot',
                     extent=[0, dataset.Lx, 0, dataset.Ly],
                     origin='lower', aspect='equal')
    ax6.set_xlabel('x (m)')
    ax6.set_ylabel('y (m)')
    ax6.set_title('c3) RMS Energy Distribution (4x Interpolated)')
    plt.colorbar(im6, ax=ax6, label='RMS Energy')
    
    # ===== d) Defect label images =====
    ax7 = plt.subplot(3, 3, 7)
    im7 = ax7.imshow(img, cmap='hot', vmin=0, vmax=1)
    ax7.set_title('d1) Defect Label Image')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, label='Intensity')
    
    # d2: 3D defect image
    ax8 = plt.subplot(3, 3, 8, projection='3d')
    x_img = np.linspace(0, 1, img.shape[1])
    y_img = np.linspace(0, 1, img.shape[0])
    X_img, Y_img = np.meshgrid(x_img, y_img)
    surf2 = ax8.plot_surface(X_img, Y_img, img, cmap='hot', alpha=0.9)
    ax8.set_xlabel('x (normalized)')
    ax8.set_ylabel('y (normalized)')
    ax8.set_zlabel('Intensity')
    ax8.set_title('d2) 3D Defect Distribution')
    
    # ===== 【新增】d3: Wavenumber-domain Imaging (波数域成像) =====
    ax9 = plt.subplot(3, 3, 9)
    # 2D FFT波数域成像
    kspace = np.fft.fft2(spatial_snapshot)  # 2D FFT
    kspace_shifted = np.fft.fftshift(kspace)  # 频谱中心化
    kspace_mag = np.abs(kspace_shifted)  # 幅值
    
    # 【修改】先对幅值进行4倍插值
    kspace_mag_interp = zoom(kspace_mag, 4, order=1)  # 4x线性插值
    
    # 然后对插值后的结果取对数
    kspace_log = np.log10(kspace_mag_interp + 1e-10)
    
    im9 = ax9.imshow(kspace_log, cmap='viridis', origin='lower', aspect='equal')
    ax9.set_xlabel('kx (wavenumber)')
    ax9.set_ylabel('ky (wavenumber)')
    ax9.set_title('d3) Wavenumber-domain Imaging (2D FFT, 4x Interpolated, log scale)')
    plt.colorbar(im9, ax=ax9, label='log10(Magnitude)')
    
    plt.suptitle('3D Spatiotemporal Ultrasonic Dataset - Complete Visualization', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Create images directory if not exists
    os.makedirs('images/dataset_check', exist_ok=True)
    plt.savefig('images/dataset_check/dataset_3d_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to images/dataset_check/dataset_3d_visualization.png")
    plt.show()
    
    print("✅ Visualization complete!\n")


def test_simple_dataset_basic():
    """测试简化数据集基本功能"""
    print("=" * 60)
    print("Test: Simple Dataset Basic Functionality")
    print("=" * 60)
    
    dataset = SimpleUSDataset3D(
        n_samples=10,
        nx=5,
        ny=5,
        sig_len=50,
        img_size=10,
        precompute=True
    )
    
    # 检查长度
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
    print(f"✓ Dataset length: {len(dataset)}")
    
    # 检查单个样本
    sig, img = dataset[0]
    assert sig.shape == (5, 5, 50), f"Signal shape error: {sig.shape}"
    assert img.shape == (10, 10), f"Image shape error: {img.shape}"
    print(f"✓ Sample shapes: signal{sig.shape}, image{img.shape}")
    
    # 检查值域
    assert sig.min() >= -1 and sig.max() <= 1, "Signal values out of range"
    assert img.min() >= 0 and img.max() <= 1, "Image values out of range"
    print(f"✓ Value ranges: sig[{sig.min():.3f}, {sig.max():.3f}], img[{img.min():.3f}, {img.max():.3f}]")
    
    # 数据集信息
    info = dataset.get_info()
    print(f"✓ Dataset info: {info}")
    
    print("✅ Simple dataset test passed!\n")


def visualize_simple_dataset():
    """可视化简化数据集"""
    print("=" * 60)
    print("Visualization: Simple Dataset Analysis")
    print("=" * 60)
    
    # 创建数据集
    dataset = SimpleUSDataset3D(
        n_samples=1,
        nx=5,
        ny=5,
        sig_len=50,
        img_size=10,
        max_defects=2,
        precompute=True
    )
    
    sig, img = dataset[0]
    print(f"Data shapes: signal{sig.shape}, image{img.shape}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 5))
    
    # ===== 1. 某个点的时域波形 =====
    ax1 = plt.subplot(1, 3, 1)
    sample_y, sample_x = 2, 2  # 中心点
    time_signal = sig[sample_y, sample_x, :]
    t_vec = np.linspace(0, dataset.T, dataset.sig_len)
    ax1.plot(t_vec * 1e6, time_signal, linewidth=1.2, color='steelblue')
    ax1.set_xlabel('Time (μs)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Time Signal at Point ({sample_x}, {sample_y})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    
    # ===== 2. 空间分布 (某时刻) =====
    ax2 = plt.subplot(1, 3, 2)
    time_idx = 8  # 中间时刻
    spatial_snapshot = sig[:, :, time_idx]
    
    # 4倍插值
    from scipy.ndimage import zoom
    spatial_interp = zoom(spatial_snapshot, 4, order=1)
    
    im2 = ax2.imshow(spatial_interp, cmap='seismic',
                     extent=[0, dataset.L * 1000, 0, dataset.L * 1000],
                     origin='lower', aspect='equal',
                     vmin=-1, vmax=1)
    ax2.set_xlabel('x (mm)', fontsize=11)
    ax2.set_ylabel('y (mm)', fontsize=11)
    ax2.set_title(f'Spatial Distribution at t={t_vec[time_idx]*1e6:.1f}μs', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Amplitude', shrink=0.8)
    
    # 标记激励源位置
    ax2.plot(dataset.src_x * 1000, dataset.src_y * 1000, 'g*', markersize=15, label='Source')
    # 标记测点
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
    
    # 保存图像
    os.makedirs('images/dataset_check', exist_ok=True)
    plt.savefig('images/dataset_check/dataset_simple_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to images/dataset_check/dataset_simple_visualization.png")
    plt.show()
    
    print("✅ Simple dataset visualization complete!\n")


def test_cropped_dataset():
    """测试裁剪数据集功能"""
    print("=" * 60)
    print("Test: Cropped Dataset Functionality")
    print("=" * 60)
    
    # 创建原始数据集
    base_dataset = SimpleUSDataset3D(
        n_samples=10,
        nx=5,
        ny=5,
        sig_len=100,
        img_size=10,
        precompute=True
    )
    
    # 测试边界模式
    print("\n--- Testing boundary mode ---")
    cropped_dataset, cropper = create_cropped_dataset(
        base_dataset,
        crop_mode='boundary',
        random_per_sample=False
    )
    
    sig_cropped, img = cropped_dataset[0]
    print(f"✓ Cropped signal shape: {sig_cropped.shape}")
    print(f"✓ Image shape (unchanged): {img.shape}")
    print(f"✓ Branch dim: {cropped_dataset.get_branch_dim()}")
    
    assert sig_cropped.shape[0] == 16, "Boundary mode should keep 16 points"
    assert sig_cropped.shape[1] == 100, "Time steps unchanged"
    assert img.shape == (10, 10), "Image shape unchanged"
    
    # 测试随机模式
    print("\n--- Testing random mode ---")
    cropped_dataset_rand, cropper_rand = create_cropped_dataset(
        base_dataset,
        crop_mode='random',
        n_keep=12,
        random_per_sample=True,
        random_seed=42
    )
    
    sig_rand, _ = cropped_dataset_rand[0]
    print(f"✓ Random cropped signal shape: {sig_rand.shape}")
    assert sig_rand.shape[0] == 12, "Should keep 12 points"
    
    print("\n✅ Cropped dataset test passed!\n")


def visualize_crop_pattern():
    """可视化裁剪模式"""
    print("=" * 60)
    print("Visualization: Crop Pattern")
    print("=" * 60)
    
    # 创建裁剪器
    cropper_boundary = SpatialCropper(
        nx=5, ny=5, sig_len=100,
        crop_mode='boundary'
    )
    
    cropper_random = SpatialCropper(
        nx=5, ny=5, sig_len=100,
        crop_mode='random',
        n_keep=12,
        random_seed=42
    )
    
    # 生成掩码
    mask_boundary = cropper_boundary.visualize_crop_pattern()
    mask_random = cropper_random.visualize_crop_pattern()
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 边界模式
    ax1 = axes[0]
    im1 = ax1.imshow(mask_boundary, cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_title('Boundary Mode (16 sensors)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x index')
    ax1.set_ylabel('y index')
    
    # 标注保留点
    for y in range(5):
        for x in range(5):
            if mask_boundary[y, x] == 1:
                ax1.plot(x, y, 'go', markersize=15)
                ax1.text(x, y, '✓', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=12)
            else:
                ax1.plot(x, y, 'rx', markersize=12, markeredgewidth=2)
    
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Kept (1) / Removed (0)')
    
    # 随机模式
    ax2 = axes[1]
    im2 = ax2.imshow(mask_random, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('Random Mode (12 sensors)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x index')
    ax2.set_ylabel('y index')
    
    for y in range(5):
        for x in range(5):
            if mask_random[y, x] == 1:
                ax2.plot(x, y, 'go', markersize=15)
                ax2.text(x, y, '✓', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=12)
            else:
                ax2.plot(x, y, 'rx', markersize=12, markeredgewidth=2)
    
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Kept (1) / Removed (0)')
    
    plt.suptitle('Spatial Cropping Patterns for DeepONet', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs('images/dataset_check', exist_ok=True)
    plt.savefig('images/dataset_check/crop_pattern_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to images/dataset_check/crop_pattern_visualization.png")
    plt.show()
    
    print("✅ Crop pattern visualization complete!\n")


def visualize_cropped_sample():
    """可视化裁剪前后的信号对比"""
    print("=" * 60)
    print("Visualization: Before/After Cropping")
    print("=" * 60)
    
    # 创建数据集
    base_dataset = SimpleUSDataset3D(
        n_samples=1,
        nx=5, ny=5, sig_len=100,
        img_size=10,
        precompute=True
    )
    
    cropped_dataset, cropper = create_cropped_dataset(
        base_dataset,
        crop_mode='boundary',
        random_per_sample=False
    )
    
    # 获取数据
    sig_full, img = base_dataset[0]  # (5, 5, 100)
    sig_cropped, _ = cropped_dataset[0]  # (16, 100)
    
    # 创建图形
    fig = plt.figure(figsize=(18, 10))
    
    # === 1. 完整信号（某时刻空间分布）===
    ax1 = plt.subplot(2, 3, 1)
    time_idx = 20
    spatial_full = sig_full[:, :, time_idx]
    
    from scipy.ndimage import zoom
    spatial_full_interp = zoom(spatial_full, 8, order=1)
    
    im1 = ax1.imshow(spatial_full_interp, cmap='seismic',
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal', vmin=-1, vmax=1)
    ax1.set_title('Full Signal (5×5 sensors)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 标记所有传感器
    x_pos = np.linspace(0, 100, 5)
    y_pos = np.linspace(0, 100, 5)
    xv, yv = np.meshgrid(x_pos, y_pos)
    ax1.plot(xv.flatten(), yv.flatten(), 'ko', markersize=6, label='All sensors')
    ax1.legend(fontsize=9)
    
    # === 2. 裁剪模式掩码 ===
    ax2 = plt.subplot(2, 3, 2)
    mask = cropper.visualize_crop_pattern()
    im2 = ax2.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('Crop Pattern (16 boundary sensors)', fontsize=12, fontweight='bold')
    
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
    
    # === 3. 损伤图（目标不变）===
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(img, cmap='hot', vmin=0, vmax=1,
                     extent=[0, 100, 0, 100],
                     origin='lower', aspect='equal')
    ax3.set_title('Target (unchanged)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # === 4. 完整信号时域波形（中心点）===
    ax4 = plt.subplot(2, 3, 4)
    t_vec = np.linspace(0, 100, 100)  # 假设100μs
    center_sig = sig_full[2, 2, :]
    ax4.plot(t_vec, center_sig, linewidth=1.2, label='Center (2,2)')
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Full Signal - Center Point', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # === 5. 裁剪信号时域波形（选择几个边界点）===
    ax5 = plt.subplot(2, 3, 5)
    kept_indices = cropper.boundary_indices
    
    # 绘制前4个边界点
    for i in range(min(4, len(kept_indices))):
        y_idx, x_idx = kept_indices[i]
        original_sig = sig_full[y_idx, x_idx, :]
        ax5.plot(t_vec, original_sig, linewidth=1.0, 
                label=f'Boundary ({x_idx},{y_idx})', alpha=0.8)
    
    ax5.set_xlabel('Time (μs)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('Cropped Signals - Boundary Points', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    
    # === 6. 维度对比 ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    text_info = f"""
    📊 Dimension Comparison
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔵 Full Dataset:
       • Signal shape: (5, 5, 100)
       • Flattened: {5*5*100} dims
       • Branch input: {5*5*100}
    
    🔪 Cropped Dataset:
       • Signal shape: (16, 100)
       • Flattened: {16*100} dims
       • Branch input: {16*100}
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📉 Dimension reduction:
       {5*5*100} → {16*100}
       ({16*100/2500*100:.1f}% of original)
    
    ✅ Target unchanged:
       (10, 10) = 100 output points
    """
    
    ax6.text(0.1, 0.5, text_info, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    
    plt.suptitle('Spatial Cropping: Input Reduction with Unchanged Target', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs('images/dataset_check', exist_ok=True)
    plt.savefig('images/dataset_check/cropped_sample_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to images/dataset_check/cropped_sample_visualization.png")
    plt.show()
    
    print("✅ Cropped sample visualization complete!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Dataset Module Complete Tests - Including 3D Support")
    print("=" * 60 + "\n")
    
    try:
        test_dataset_3d_basic()
        test_spectrogram_3d()
        
        # Visualization test
        visualize_3d_dataset()
        
        # 简化数据集测试
        test_simple_dataset_basic()
        visualize_simple_dataset()
        
        # 【新增】裁剪功能测试
        test_cropped_dataset()
        visualize_crop_pattern()
        visualize_cropped_sample()
        
        print("=" * 60)
        print("🎉 All tests passed! 3D dataset integration successful!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Exception error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
