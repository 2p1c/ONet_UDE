"""
3D时空超声数据集

物理建模:
- 输入: f(x, y, t) 时空振动信号
- 两种Lamb波模态: S0 (快速), A0 (慢速)
- 包含直接波和缺陷散射
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from scipy import signal as scipy_signal


def batch_spectrogram_3d(
    sig_3d: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 8,
    window: str = 'hann'
) -> torch.Tensor:
    """
    计算3D信号的短时傅里叶变换(STFT)频谱图
    
    Args:
        sig_3d: 输入信号，shape (batch, ny, nx, sig_len) 或 (ny, nx, sig_len)
        n_fft: FFT点数
        hop_length: 帧移
        window: 窗函数类型
    
    Returns:
        频谱图张量，shape (batch, ny, nx, n_freq, n_frames)
    """
    # 确保输入是4D (batch, ny, nx, sig_len)
    if sig_3d.ndim == 3:
        sig_3d = sig_3d[np.newaxis, ...]
    
    batch, ny, nx, sig_len = sig_3d.shape
    
    # 先用一个样本计算STFT，获取实际输出尺寸
    f, t, Zxx_sample = scipy_signal.stft(
        sig_3d[0, 0, 0, :],
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window=window
    )
    n_freq, n_frames = Zxx_sample.shape
    
    # 初始化输出（使用实际尺寸）
    spec_batch = np.zeros((batch, ny, nx, n_freq, n_frames), dtype=np.float32)
    
    # 对每个batch、每个空间点计算STFT
    for b in range(batch):
        for i in range(ny):
            for j in range(nx):
                signal_1d = sig_3d[b, i, j, :]
                # 使用scipy计算STFT
                _, _, Zxx = scipy_signal.stft(
                    signal_1d,
                    nperseg=n_fft,
                    noverlap=n_fft - hop_length,
                    window=window
                )
                # 取幅值的对数（避免数值问题）
                spec_batch[b, i, j, :, :] = np.log10(np.abs(Zxx) + 1e-10)
    
    return torch.from_numpy(spec_batch)


class ToyUSDataset3D(Dataset):
    """
    中等保真时空（f(x,y,t)）玩具超声数据集
    - 输出 X: 时空信号 array, shape (ny, nx, sig_len)
    - 输出 Y: 缺陷图像 array, shape (img_size, img_size), 值为 [0,1] 的缺陷强度
    物理/建模假设见注释。
    """

    def __init__(
        self,
        n_samples: int = 1000,
        nx: int = 51,
        ny: int = 51,
        sig_len: int = 1000,
        T: float = 0.0005,  # 时间窗口(秒)，可调整
        f0: float = 200e3,  # 中心频率 200 kHz
        fs: Optional[float] = 6.25e6,  # 采样频率，如果 None 则由 sig_len/T 确定
        plate_thickness: float = 2e-3,  # 2 mm
        material: str = "aluminum",
        # 两个模态群速度（toy默认，可调整以匹配真实实验）
        v_S0: float = 5000.0,
        v_A0: float = 1500.0,
        # 衰减因子（距离相关）
        alpha_S0: float = 0.0005,
        alpha_A0: float = 0.001,
        # 缺陷参数
        defect_types: List[str] = None,
        min_defects: int = 1,
        max_defects: int = 3,
        defect_diam_range_mm: Tuple[float, float] = (10.0, 20.0),  # mm
        img_size: Optional[int] = None,
        noise_std: float = 0.01,
        # 激励位置 (normalized 0..1 in x,y), default center
        source_pos: Tuple[float, float] = (3, 3),
        # whether to pre-generate all data
        precompute: bool = True,
        # 【改进17】新增数据增强开关
        augment: bool = False,
    ):
        self.n_samples = n_samples
        self.nx = nx
        self.ny = ny
        self.sig_len = sig_len
        self.T = T
        self.f0 = f0
        self.fs = fs if fs is not None else (sig_len / T)
        self.dt = 1.0 / self.fs
        self.t_vec = np.linspace(0, T, sig_len)
        self.plate_thickness = plate_thickness
        self.material = material

        self.v_S0 = v_S0
        self.v_A0 = v_A0
        self.alpha_S0 = alpha_S0
        self.alpha_A0 = alpha_A0

        self.defect_types = (
            defect_types if defect_types is not None else ["hole", "delam"]
        )
        self.min_defects = min_defects
        self.max_defects = max_defects
        self.defect_diam_range_mm = defect_diam_range_mm

        self.img_size = img_size if img_size is not None else max(nx, ny)
        self.noise_std = noise_std

        self.source_pos = source_pos  # normalized coordinates (0..1)
        self.precompute = precompute
        # 【改进17】新增数据增强开关
        self.augment = augment

        # prepare grid (normalized coordinates 0..1, then map to meters if needed)
        # For toy we assume unit plate size LxL with L = 0.41 m by default mapping to your experiments
        self.Lx = 0.1  # meters (you can change mapping scale)
        self.Ly = 0.1

        # spatial sampling coordinates (in meters)
        xs = np.linspace(0, self.Lx, self.nx)
        ys = np.linspace(0, self.Ly, self.ny)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        self.xv = xv
        self.yv = yv
        # source position in meters
        self.src_x = self.source_pos[0] * self.Lx
        self.src_y = self.source_pos[1] * self.Ly

        # storage
        self.signals: List[np.ndarray] = []
        self.images: List[np.ndarray] = []

        if self.precompute:
            self._generate_all_data()

    def _generate_all_data(self):
        self.signals = []
        self.images = []
        for _ in range(self.n_samples):
            sig_tensor, img = self._generate_sample()
            self.signals.append(sig_tensor.astype(np.float32))
            self.images.append(img.astype(np.float32))

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成单个样本：
        - img: 2D 缺陷强度图 (img_size, img_size)
        - sig_tensor: 时空信号 (ny, nx, sig_len)
        """
        # 1) 生成随机缺陷列表（位置为 normalized 0..1）
        k = random.randint(self.min_defects, self.max_defects)
        defects = []
        for _ in range(k):
            dtype = random.choice(self.defect_types)
            x_norm = random.uniform(0.1, 0.9)
            y_norm = random.uniform(0.1, 0.9)
            diam_mm = random.uniform(*self.defect_diam_range_mm)
            defects.append({"type": dtype, "x": x_norm, "y": y_norm, "d_mm": diam_mm})

        # 2) 构建图像标签（像素坐标系）
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        xv_img, yv_img = np.meshgrid(
            np.linspace(0, 1, self.img_size),
            np.linspace(0, 1, self.img_size),
            indexing="xy",
        )
        for d in defects:
            cx = d["x"]
            cy = d["y"]
            # convert diam mm -> normalized radius relative to plate size (use Lx as reference)
            # assume Lx meters => diam_m = d_mm / 1000
            diam_m = d["d_mm"] / 1000.0
            # normalized radius in [0,1] over plate length
            r_norm = (diam_m / 2.0) / max(self.Lx, self.Ly)
            # Gaussian blob
            sigma = r_norm * 0.5  # control spread (smaller sigma -> sharper)
            g = np.exp(
                -((xv_img - cx) ** 2 + (yv_img - cy) ** 2) / (2 * sigma**2 + 1e-12)
            )
            # scale by diameter to reflect larger defects stronger
            intensity = (d["d_mm"] - self.defect_diam_range_mm[0]) / (
                self.defect_diam_range_mm[1] - self.defect_diam_range_mm[0]
            ) + 0.2
            img += intensity * g

        # normalize image to [0,1]
        img = img / (img.max() + 1e-12)
        img = np.clip(img, 0.0, 1.0)

        # 3) 计算包络参数（基于中心频率）
        T0 = 1.0 / self.f0  # 中心频率的周期
        
        # 直达波：5个周期的半峰全宽 (FWHM)
        # FWHM = 2.355 * σ，因此 σ = FWHM / 2.355
        direct_fwhm = 5 * T0
        direct_sigma = direct_fwhm / 2.355  # ≈ 2.12 * T0
        
        # 3) 生成时空信号：对每个测点叠加直接波与缺陷散射
        sig_tensor = np.zeros((self.ny, self.nx, self.sig_len), dtype=np.float32)

        # flattened defect positions in meters
        defect_positions_m = []
        for d in defects:
            dx = d["x"] * self.Lx
            dy = d["y"] * self.Ly
            defect_positions_m.append((d, dx, dy))

        # precompute some constants
        omega0 = 2 * np.pi * self.f0

        for iy in range(self.ny):
            for ix in range(self.nx):
                rx = self.xv[iy, ix]
                ry = self.yv[iy, ix]

                # initialize signal
                s = np.zeros(self.sig_len, dtype=np.float64)

                # ---- 3a: 直接波（source -> receiver） for both modes ----
                dist_sr = (
                    np.sqrt((rx - self.src_x) ** 2 + (ry - self.src_y) ** 2) + 1e-12
                )
                # S0 直达波 - 使用5个周期的包络宽度
                tau_s = dist_sr / self.v_S0
                geo_sr = 1.0 / np.sqrt(dist_sr + 1e-6)
                env_s = geo_sr * np.exp(-self.alpha_S0 * dist_sr) * np.exp(
                    -((self.t_vec - tau_s) ** 2) / (2 * 3 * direct_sigma ** 2)
                )
                carrier_s = np.sin(2 * np.pi * self.f0 * (self.t_vec - tau_s))
                s += 1.0 * env_s * carrier_s  # 直达波 S0

                # A0 直达波 - 使用5个周期的包络宽度
                tau_a = dist_sr / self.v_A0
                env_a = geo_sr * np.exp(-self.alpha_A0 * dist_sr) * np.exp(
                    -((self.t_vec - tau_a) ** 2) / (2 * direct_sigma ** 2)
                )
                carrier_a = np.sin(2 * np.pi * self.f0 * (self.t_vec - tau_a))
                s += 0.9 * env_a * carrier_a  # 直达波 A0

                # ---- 3b: 缺陷散射 (source -> defect -> receiver) ----
                for dmeta, dx_m, dy_m in defect_positions_m:
                    # path lengths
                    dist_sd = (
                        np.sqrt((dx_m - self.src_x) ** 2 + (dy_m - self.src_y) ** 2)
                        + 1e-12
                    )
                    dist_dr = np.sqrt((rx - dx_m) ** 2 + (ry - dy_m) ** 2) + 1e-12
                    path_total = dist_sd + dist_dr

                    # scattering delay approximated by path_total / mode_velocity
                    tau_scat_s = path_total / self.v_S0
                    tau_scat_a = path_total / self.v_A0

                    # defect-dependent scattering amplitude
                    diam = dmeta["d_mm"]
                    diam_norm = (diam - self.defect_diam_range_mm[0]) / (
                        self.defect_diam_range_mm[1] - self.defect_diam_range_mm[0]
                    )
                    # 散射振幅：大缺陷散射更强
                    base_amp = 0.15 + 0.50 * diam_norm  # 0.15~0.65

                    # type-dependent factor
                    if dmeta["type"] == "hole":
                        type_factor = 1.0
                        phase_shift = 0.0
                    else:  # delamination
                        type_factor = 0.7
                        phase_shift = np.pi / 6

                    # 散射损失因子：适度衰减
                    scattering_loss = 0.5 + 0.3 * diam_norm  # 0.5~0.8
                    
                    # 散射波包络宽度：5个周期 FWHM + 缺陷相关展宽
                    # 小缺陷：展宽到 7个周期；大缺陷：保持 5个周期
                    scatter_fwhm = 5 * T0 + (2 * T0) * (1 - diam_norm)  # 5~7 周期
                    scatter_sigma = scatter_fwhm / 2.355
                    
                    # S0 scattered piece（系数 0.35）
                    geo_scat = 1.0 / np.sqrt(dist_sd * dist_dr + 1e-6)
                    env_ss = geo_scat * np.exp(-self.alpha_S0 * path_total) * np.exp(
                        -((self.t_vec - tau_scat_s) ** 2) / (2 * 3 * scatter_sigma ** 2)
                    )
                    carrier_ss = np.sin(
                        2 * np.pi * self.f0 * (self.t_vec - tau_scat_s) + phase_shift
                    )
                    s += base_amp * type_factor * 0.35 * scattering_loss * env_ss * carrier_ss

                    # A0 scattered piece（系数 0.40，稍宽的包络）
                    scatter_fwhm_a0 = 5 * T0 + (3 * T0) * (1 - diam_norm)  # 5~8 周期
                    scatter_sigma_a0 = scatter_fwhm_a0 / 2.355
                    
                    env_sa = geo_scat * np.exp(-self.alpha_A0 * path_total) * np.exp(
                        -((self.t_vec - tau_scat_a) ** 2) / (2 * scatter_sigma_a0 ** 2)
                    )
                    carrier_sa = np.sin(
                        2 * np.pi * self.f0 * (self.t_vec - tau_scat_a) + phase_shift
                    )
                    s += base_amp * type_factor * 0.40 * scattering_loss * env_sa * carrier_sa

                # ---- 3c: 噪声与归一化 ----
                s += np.random.normal(scale=self.noise_std, size=s.shape)
                # normalize per-receiver to keep amplitude in [-1,1]
                smax = np.max(np.abs(s)) + 1e-12
                s = s / smax

                sig_tensor[iy, ix, :] = s

        return sig_tensor, img

    def _augment_signal(self, sig_3d: np.ndarray) -> np.ndarray:
        """
        【改进18】数据增强 - 随机噪声扰动
        
        策略:
        - 随机高斯噪声
        - 随机幅度缩放
        - 随机时间平移(循环移位)
        """
        if not self.augment:
            return sig_3d
        
        sig_aug = sig_3d.copy()
        
        # 1. 随机噪声 (±10%)
        if np.random.rand() > 0.5:
            noise_level = np.random.uniform(0.01, 0.03)
            sig_aug += np.random.randn(*sig_3d.shape) * noise_level
        
        # 2. 随机幅度缩放 (0.9-1.1)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            sig_aug *= scale
        
        # 3. 随机时间平移 (沿时间轴循环移位)
        if np.random.rand() > 0.5:
            shift = np.random.randint(-20, 20)
            sig_aug = np.roll(sig_aug, shift, axis=2)
        
        # 重新归一化到[-1, 1]
        sig_aug = np.clip(sig_aug, -1, 1)
        
        return sig_aug

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        """获取样本"""
        if self.precompute:
            sig_3d, img = self.signals[idx], self.images[idx]
        else:
            sig_3d, img = self._generate_sample()
        
        # 【应用增强】
        sig_3d = self._augment_signal(sig_3d)
        
        return sig_3d, img

    def get_info(self):
        return {
            "n_samples": self.n_samples,
            "nx": self.nx,
            "ny": self.ny,
            "sig_len": self.sig_len,
            "T": self.T,
            "f0": self.f0,
            "v_S0": self.v_S0,
            "v_A0": self.v_A0,
            "defect_range_mm": self.defect_diam_range_mm,
            "img_size": self.img_size,
        }