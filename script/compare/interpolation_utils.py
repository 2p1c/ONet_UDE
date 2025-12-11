"""
插值工具函数模块

提供空间插值和输出插值功能
"""

import numpy as np
from scipy.ndimage import zoom
from PIL import Image


def interpolate_spatial(sig_3d, target_shape):
    """
    空间插值 (双线性)
    
    Args:
        sig_3d: (ny_in, nx_in, sig_len) - 输入时空信号
        target_shape: (ny_out, nx_out) - 目标空间尺寸
    
    Returns:
        (ny_out, nx_out, sig_len) - 插值后的信号
    
    Example:
        sig_in = np.random.rand(3, 3, 100)
        sig_out = interpolate_spatial(sig_in, (5, 5))
        assert sig_out.shape == (5, 5, 100)
    """
    if not isinstance(sig_3d, np.ndarray):
        raise TypeError(f"sig_3d must be numpy array, got {type(sig_3d)}")
    
    if sig_3d.ndim != 3:
        raise ValueError(f"sig_3d must be 3D array, got shape {sig_3d.shape}")
    
    zoom_factors = (
        target_shape[0] / sig_3d.shape[0],  # y方向
        target_shape[1] / sig_3d.shape[1],  # x方向
        1.0  # 时间维度不变
    )
    
    return zoom(sig_3d, zoom_factors, order=1)  # order=1为双线性插值


def interpolate_output_image(img, target_size):
    """
    输出图像插值 (双三次)
    
    Args:
        img: (H, W) - 输入图像, 值域[0,1]
        target_size: int - 目标尺寸
    
    Returns:
        (target_size, target_size) - 插值后的图像
    
    Example:
        img_in = np.random.rand(10, 10)
        img_out = interpolate_output_image(img_in, 20)
        assert img_out.shape == (20, 20)
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be numpy array, got {type(img)}")
    
    if img.ndim != 2:
        raise ValueError(f"img must be 2D array, got shape {img.shape}")
    
    # 转换为uint8 (PIL需要)
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # 使用PIL进行双三次插值
    img_pil = Image.fromarray(img_uint8)
    img_resized = img_pil.resize((target_size, target_size), Image.BICUBIC)
    
    # 转回float32并归一化
    return np.array(img_resized, dtype=np.float32) / 255.0


def interpolate_temporal(sig_3d, target_len):
    """
    时间插值 (立方插值)
    
    Args:
        sig_3d: (ny, nx, sig_len_in) - 输入时空信号
        target_len: int - 目标时间长度
    
    Returns:
        (ny, nx, target_len) - 插值后的信号
    
    Example:
        sig_in = np.random.rand(5, 5, 50)
        sig_out = interpolate_temporal(sig_in, 100)
        assert sig_out.shape == (5, 5, 100)
    """
    if not isinstance(sig_3d, np.ndarray):
        raise TypeError(f"sig_3d must be numpy array, got {type(sig_3d)}")
    
    if sig_3d.ndim != 3:
        raise ValueError(f"sig_3d must be 3D array, got shape {sig_3d.shape}")
    
    from scipy.interpolate import interp1d
    
    ny, nx, sig_len_in = sig_3d.shape
    sig_out = np.zeros((ny, nx, target_len), dtype=np.float32)
    
    t_in = np.linspace(0, 1, sig_len_in)
    t_out = np.linspace(0, 1, target_len)
    
    for i in range(ny):
        for j in range(nx):
            interp_func = interp1d(t_in, sig_3d[i, j, :], kind='cubic')
            sig_out[i, j, :] = interp_func(t_out)
    
    return sig_out
