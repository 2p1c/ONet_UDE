"""
@author: jpzxshi
"""

import torch
import torch.nn as nn

from .module import StructureNN
from .fnn import FNN


class DeepONet(StructureNN):
    """Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    """

    def __init__(
        self,
        branch_dim,
        trunk_dim,
        branch_depth=2,
        trunk_depth=3,
        width=50,
        activation="relu",
        initializer="Glorot normal",
        dropout=0.1,  # 【新增】Dropout率
    ):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.dropout = dropout  # 【新增】

        self.modus = self.__init_modules()
        self.params = self.__init_params()
        self.__initialize()

    def forward(self, x):
        x_branch, x_trunk = x[..., : self.branch_dim], x[..., self.branch_dim :]
        x_branch = self.modus["Branch"](x_branch)

        # 【新增】Branch 输出后添加 Dropout
        if self.training and self.dropout > 0:
            x_branch = self.modus["Dropout"](x_branch)

        for i in range(1, self.trunk_depth):
            x_trunk = self.modus["TrActM{}".format(i)](
                self.modus["TrLinM{}".format(i)](x_trunk)
            )
            # 【新增】Trunk 每层后添加 Dropout
            if self.training and self.dropout > 0 and i < self.trunk_depth - 1:
                x_trunk = self.modus["Dropout"](x_trunk)

        # 关键创新
        return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params["bias"]

    def __init_modules(self):
        # branch net
        modules = nn.ModuleDict()
        modules["Branch"] = FNN(
            self.branch_dim,
            self.width,
            self.branch_depth,
            self.width,
            self.activation,
            self.initializer,
        )

        # 【新增】Dropout 层
        if self.dropout > 0:
            modules["Dropout"] = nn.Dropout(self.dropout)

        # trunk net
        # 第1层
        modules["TrLinM1"] = nn.Linear(self.trunk_dim, self.width)
        modules["TrActM1"] = self.Act
        # 第2到trunk_depth-1层
        for i in range(2, self.trunk_depth):
            modules["TrLinM{}".format(i)] = nn.Linear(self.width, self.width)
            modules["TrActM{}".format(i)] = self.Act
        return modules

    def __init_params(self):
        params = nn.ParameterDict()
        params["bias"] = nn.Parameter(torch.zeros([1]))
        return params

    def __initialize(self):
        for i in range(1, self.trunk_depth):
            self.weight_init_(self.modus["TrLinM{}".format(i)].weight)
            nn.init.constant_(self.modus["TrLinM{}".format(i)].bias, 0)
