"""
简化版3D时空超声数据集 - 用于DeepONet快速验证

输入: 5×5空间网格 × 50时间步
输出: 10×10损伤概率图

物理建模:
- 双模态Lamb波 (S0/A0)
- 距离衰减 + 损伤散射
- 高信噪比
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
import random


class SimpleUSDataset3D(Dataset):
    """
    简化版时空超声数据集

    输出:
    - X: 时空信号 (5, 5, 50) - 5×5空间测点，每点50个时间步
    - Y: 损伤概率图 (10, 10) - 值域 [0, 1]
    """

    def __init__(
        self,
        n_samples: int = 1000,
        nx: int = 5,  # 空间网格x方向
        ny: int = 5,  # 空间网格y方向
        sig_len: int = 100,  # 时间步数
        img_size: int = 10,  # 损伤图尺寸
        # 物理参数
        T: float = 0.0001,  # 时间窗口 100us
        f0: float = 200e3,  # 中心频率 200kHz
        v_S0: float = 5000.0,  # S0波速 m/s
        v_A0: float = 1500.0,  # A0波速 m/s
        plate_size: float = 0.1,  # 板尺寸 0.1m (100mm)
        # 损伤参数
        min_defects: int = 1,
        max_defects: int = 5,
        defect_intensity_range: Tuple[float, float] = (0.3, 0.8),
        # 噪声
        noise_std: float = 0.02,  # 2% 噪声，保证高SNR
        # 激励位置 (归一化坐标 0-1)
        source_pos: Tuple[float, float] = (1.3, 1.3),  # 中心激励
        precompute: bool = True,
    ):
        self.n_samples = n_samples
        self.nx = nx
        self.ny = ny
        self.sig_len = sig_len
        self.img_size = img_size

        # 物理参数
        self.T = T
        self.f0 = f0
        self.v_S0 = v_S0
        self.v_A0 = v_A0
        self.L = plate_size
        self.dt = T / sig_len
        self.t_vec = np.linspace(0, T, sig_len)

        # 损伤参数
        self.min_defects = min_defects
        self.max_defects = max_defects
        self.defect_intensity_range = defect_intensity_range

        # 噪声
        self.noise_std = noise_std

        # 激励源位置 (米)
        self.src_x = source_pos[0] * self.L
        self.src_y = source_pos[1] * self.L

        # 构建空间网格 (米)
        xs = np.linspace(0, self.L, self.nx)
        ys = np.linspace(0, self.L, self.ny)
        self.xv, self.yv = np.meshgrid(xs, ys, indexing="xy")

        # 预计算
        self.precompute = precompute
        self.signals = []
        self.images = []

        if self.precompute:
            self._generate_all_data()

    def _generate_all_data(self):
        """预生成所有样本"""
        for _ in range(self.n_samples):
            sig, img = self._generate_sample()
            self.signals.append(sig.astype(np.float32))
            self.images.append(img.astype(np.float32))

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成单个样本

        Returns:
            sig: (ny, nx, sig_len) 时空信号
            img: (img_size, img_size) 损伤概率图
        """
        # 1. 生成随机损伤
        n_defects = random.randint(self.min_defects, self.max_defects)
        defects = []
        for _ in range(n_defects):
            x_norm = random.uniform(0.2, 0.8)
            y_norm = random.uniform(0.2, 0.8)
            intensity = random.uniform(*self.defect_intensity_range)
            defects.append({"x": x_norm, "y": y_norm, "intensity": intensity})

        # 2. 构建损伤概率图
        img = self._create_damage_image(defects)

        # 3. 生成时空信号
        sig = self._generate_signal(defects)

        return sig, img

    def _create_damage_image(self, defects: List[dict]) -> np.ndarray:
        """
        构建损伤概率图

        Args:
            defects: 损伤列表

        Returns:
            img: (img_size, img_size) 概率图
        """
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # 构建归一化网格
        xv_img, yv_img = np.meshgrid(
            np.linspace(0, 1, self.img_size),
            np.linspace(0, 1, self.img_size),
            indexing="xy",
        )

        # 叠加每个损伤的高斯分布
        for d in defects:
            cx, cy = d["x"], d["y"]
            intensity = d["intensity"]

            # 高斯半径 (归一化)
            sigma = 0.08  # 约占图像的8%

            # 高斯blob
            g = np.exp(-((xv_img - cx) ** 2 + (yv_img - cy) ** 2) / (2 * sigma**2))
            img += intensity * g

        # 归一化到 [0, 1]
        img = np.clip(img / (img.max() + 1e-12), 0, 1)

        return img

    def _generate_signal(self, defects: List[dict]) -> np.ndarray:
        """
        生成时空信号矩阵

        物理模型:
        1. 直达波: 激励源 → 接收点 (S0 + A0模态)
        2. 散射波: 激励源 → 损伤 → 接收点 (幅度由损伤强度调制)

        Args:
            defects: 损伤列表

        Returns:
            sig: (ny, nx, sig_len) 时空信号
        """
        sig = np.zeros((self.ny, self.nx, self.sig_len), dtype=np.float32)

        # 转换损伤坐标到米
        defect_pos_m = [
            (d["x"] * self.L, d["y"] * self.L, d["intensity"]) for d in defects
        ]

        # 包络参数
        T0 = 1.0 / self.f0
        sigma_direct = 3 * T0 / 2.355  # 直达波窄
        sigma_scatter = 5 * T0 / 2.355  # 散射波宽

        # 遍历每个接收点
        for iy in range(self.ny):
            for ix in range(self.nx):
                rx = self.xv[iy, ix]
                ry = self.yv[iy, ix]

                s = np.zeros(self.sig_len, dtype=np.float64)

                # === 1. 直达波 (Source → Receiver) ===
                dist_sr = (
                    np.sqrt((rx - self.src_x) ** 2 + (ry - self.src_y) ** 2) + 1e-12
                )

                # S0 直达波
                tau_s0 = dist_sr / self.v_S0
                amp_s0 = 1.0 / np.sqrt(dist_sr + 1e-6)  # 几何衰减
                env_s0 = amp_s0 * np.exp(
                    -((self.t_vec - tau_s0) ** 2) / (2 * sigma_direct**2)
                )
                carrier_s0 = np.sin(2 * np.pi * self.f0 * (self.t_vec - tau_s0))
                s += 1.0 * env_s0 * carrier_s0

                # A0 直达波
                tau_a0 = dist_sr / self.v_A0
                env_a0 = amp_s0 * np.exp(
                    -((self.t_vec - tau_a0) ** 2) / (2 * sigma_direct**2)
                )
                carrier_a0 = np.sin(2 * np.pi * self.f0 * (self.t_vec - tau_a0))
                s += 0.9 * env_a0 * carrier_a0  # A0振幅较小

                # === 2. 损伤散射波 (Source → Defect → Receiver) ===
                for dx_m, dy_m, intensity in defect_pos_m:
                    # 路径长度
                    dist_sd = (
                        np.sqrt((dx_m - self.src_x) ** 2 + (dy_m - self.src_y) ** 2)
                        + 1e-12
                    )
                    dist_dr = np.sqrt((rx - dx_m) ** 2 + (ry - dy_m) ** 2) + 1e-12
                    path_total = dist_sd + dist_dr

                    # 散射幅度 (强度越大，散射越强)
                    scatter_amp = 0.15 + 0.3 * intensity / np.sqrt(dist_sd * dist_dr + 1e-6)

                    # S0 散射
                    tau_s0_scatter = path_total / self.v_S0
                    env_s0_scatter = scatter_amp * np.exp(
                        -((self.t_vec - tau_s0_scatter) ** 2) / (2 * sigma_scatter**2)
                    )
                    carrier_s0_scatter = np.sin(
                        2 * np.pi * self.f0 * (self.t_vec - tau_s0_scatter)
                    )
                    s += env_s0_scatter * carrier_s0_scatter

                    # A0 散射
                    tau_a0_scatter = path_total / self.v_A0
                    env_a0_scatter = (
                        scatter_amp
                        * 0.5
                        * np.exp(
                            -((self.t_vec - tau_a0_scatter) ** 2)
                            / (2 * sigma_scatter**2)
                        )
                    )
                    carrier_a0_scatter = np.sin(
                        2 * np.pi * self.f0 * (self.t_vec - tau_a0_scatter)
                    )
                    s += env_a0_scatter * carrier_a0_scatter

                # === 3. 添加高斯噪声 ===
                s += np.random.normal(0, self.noise_std, s.shape)

                # === 4. 归一化到 [-1, 1] ===
                s_max = np.max(np.abs(s)) + 1e-12
                s = s / s_max

                sig[iy, ix, :] = s

        return sig

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取样本

        Returns:
            sig: (ny, nx, sig_len) = (5, 5, 50)
            img: (img_size, img_size) = (10, 10)
        """
        if self.precompute:
            return self.signals[idx], self.images[idx]
        else:
            return self._generate_sample()

    def get_info(self):
        """返回数据集信息"""
        return {
            "n_samples": self.n_samples,
            "signal_shape": (self.ny, self.nx, self.sig_len),
            "image_shape": (self.img_size, self.img_size),
            "time_window": self.T,
            "center_freq": self.f0,
            "S0_velocity": self.v_S0,
            "A0_velocity": self.v_A0,
            "plate_size": self.L,
            "noise_std": self.noise_std,
        }
