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
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/dataset_3d_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to images/dataset_3d_visualization.png")
    plt.show()
    
    print("✅ Visualization complete!\n")


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
