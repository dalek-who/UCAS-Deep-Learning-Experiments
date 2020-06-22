import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FNNModel(nn.Module):
    def __init__(self):
        super(FNNModel, self).__init__()
        self.linear = nn.Linear(2 * 2, 10)

    def forward(self, x):
        pass


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.cnn = nn.Sequential(
            self._conv_relu_maxpool(conv_ic=3, conv_oc=96, conv_ks=11, conv_st=4, conv_pd=0, pool_ks=3, pool_st=2),   # 卷积层1
            self._conv_relu_maxpool(conv_ic=96, conv_oc=256, conv_ks=5, conv_st=1, conv_pd=2, pool_ks=3, pool_st=2),  # 卷积层2
            self._conv_relu_maxpool(conv_ic=256, conv_oc=384, conv_ks=3, conv_st=1, conv_pd=1, use_pooling=False),  # 卷积层3
            self._conv_relu_maxpool(conv_ic=384, conv_oc=384, conv_ks=3, conv_st=1, conv_pd=1, use_pooling=False),  # 卷积层4
            self._conv_relu_maxpool(conv_ic=384, conv_oc=256, conv_ks=3, conv_st=1, conv_pd=1, pool_ks=3, pool_st=2),  # 卷积层5
        )

        self.fnn = nn.Sequential(
            self._linear_relu_dropout(linear_i=9216, linear_o=4096, dropout=0.5),
            self._linear_relu_dropout(linear_i=4096, linear_o=4096, dropout=0.5),
            torch.nn.Linear(4096, 2)
        )

    def _conv_relu_maxpool(self, conv_ic, conv_oc, conv_ks, conv_st, conv_pd, pool_ks=None, pool_st=None, use_pooling=True):
        if use_pooling:
            layers = nn.Sequential(
            nn.Conv2d(in_channels=conv_ic, out_channels=conv_oc, kernel_size=conv_ks, stride=conv_st, padding=conv_pd),  # 第1个卷积层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_ks, stride=pool_st),
        )
        else:
            layers = nn.Sequential(
            nn.Conv2d(in_channels=conv_ic, out_channels=conv_oc, kernel_size=conv_ks, stride=conv_st, padding=conv_pd),  # 第1个卷积层
            nn.ReLU(),
        )
        return layers

    def _linear_relu_dropout(self, linear_i, linear_o, dropout):
        return nn.Sequential(
            nn.Linear(linear_i, linear_o),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.fnn(x)
        return x
