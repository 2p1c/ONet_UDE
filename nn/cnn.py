"""
基础CNN网络 - 用于时空超声信号损伤识别

输入: (batch, 100, 5, 5)  # (channels=time, height=y, width=x)
输出: (batch, 1, 10, 10)  # (channels=1, height, width)

架构:
- 3层2D卷积提取特征
- 2层转置卷积上采样到目标尺寸
- Sigmoid激活输出概率图
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    简单CNN网络 - 用于损伤概率图预测
    
    数据流:
    输入: sig (5, 5, 100)  # 原始时空信号
      ↓ reshape
    (100, 5, 5)  # 时间作为通道
      ↓ Conv2D layers
    (256, 5, 5)  # 特征提取
      ↓ ConvTranspose2D
    (1, 10, 10)  # 上采样到目标尺寸
      ↓ Sigmoid
    概率图 (10, 10)  # 值域[0,1]
    """
    
    def __init__(
        self,
        input_channels: int = 100,  # 时间步数
        hidden_channels: int = 64,
        dropout: float = 0.15,
    ):
        """
        Args:
            input_channels: 输入通道数（时间步长）
            hidden_channels: 隐藏层通道数
            dropout: Dropout率
        """
        super(SimpleCNN, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # ========== 编码器 (特征提取) ==========
        # Conv1: (100, 5, 5) → (64, 5, 5)
        self.conv1 = nn.Conv2d(
            input_channels, hidden_channels,
            kernel_size=3, padding=1  # same padding保持尺寸
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout)
        
        # Conv2: (64, 5, 5) → (128, 5, 5)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels * 2,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout)
        
        # Conv3: (128, 5, 5) → (256, 5, 5)
        self.conv3 = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 4,
            kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(hidden_channels * 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout2d(dropout)
        
        # ========== 解码器 (上采样) ==========
        # ConvTranspose1: (256, 5, 5) → (128, 10, 10)
        self.deconv1 = nn.ConvTranspose2d(
            hidden_channels * 4, hidden_channels * 2,
            kernel_size=4, stride=2, padding=1  # 上采样2倍
        )
        self.bn4 = nn.BatchNorm2d(hidden_channels * 2)
        self.relu4 = nn.ReLU(inplace=True)
        
        # ConvTranspose2: (128, 10, 10) → (1, 10, 10)
        self.deconv2 = nn.ConvTranspose2d(
            hidden_channels * 2, 1,
            kernel_size=3, padding=1  # 保持尺寸
        )
        
        # 输出激活: Sigmoid保证[0,1]
        self.sigmoid = nn.Sigmoid()
        
        # 参数初始化
        self._initialize_weights()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, time, height, width) = (batch, 100, 5, 5)
        
        Returns:
            out: 输出张量 (batch, 1, 10, 10)
        """
        # ========== 编码器 ==========
        # Conv1
        x = self.conv1(x)      # (B, 64, 5, 5)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Conv2
        x = self.conv2(x)      # (B, 128, 5, 5)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Conv3
        x = self.conv3(x)      # (B, 256, 5, 5)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # ========== 解码器 ==========
        # DeConv1
        x = self.deconv1(x)    # (B, 128, 10, 10)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # DeConv2
        x = self.deconv2(x)    # (B, 1, 10, 10)
        
        # Sigmoid激活
        x = self.sigmoid(x)    # (B, 1, 10, 10) ∈ [0, 1]
        
        return x
    
    def _initialize_weights(self):
        """初始化网络权重 - 使用He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_info(self):
        """返回模型信息"""
        n_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'SimpleCNN',
            'input_shape': f'(batch, {self.input_channels}, 5, 5)',
            'output_shape': '(batch, 1, 10, 10)',
            'hidden_channels': self.hidden_channels,
            'dropout': self.dropout,
            'total_parameters': n_params,
        }
