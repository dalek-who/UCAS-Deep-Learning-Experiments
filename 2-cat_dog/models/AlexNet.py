import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class AlexNet(nn.Module):
    def __init__(self, tags_num: int=2, fixed_cnn_param: bool=False):
        super(AlexNet, self).__init__()
        self.fixed_cnn_param = fixed_cnn_param
        self.cnn = nn.Sequential(
            self._conv_relu_maxpool(conv_ic=3, conv_oc=64, conv_ks=11, conv_st=4, conv_pd=0, pool_ks=3, pool_st=2),
            self._conv_relu_maxpool(conv_ic=64, conv_oc=192, conv_ks=5, conv_st=1, conv_pd=2, pool_ks=3, pool_st=2),
            self._conv_relu_maxpool(conv_ic=192, conv_oc=384, conv_ks=3, conv_st=1, conv_pd=1, use_pooling=False),
            self._conv_relu_maxpool(conv_ic=384, conv_oc=256, conv_ks=3, conv_st=1, conv_pd=1, use_pooling=False),
            self._conv_relu_maxpool(conv_ic=256, conv_oc=256, conv_ks=3, conv_st=1, conv_pd=1, pool_ks=3, pool_st=2),
        )

        self.fnn = nn.Sequential(
            self._linear_relu_dropout(linear_i=9216, linear_o=4096, dropout=0.5),
            self._linear_relu_dropout(linear_i=4096, linear_o=4096, dropout=0.5),
            torch.nn.Linear(4096, tags_num)
        )

        if self.fixed_cnn_param:
            for name, param in self.named_parameters():
                if name.startswith("cnn"):
                    param.requires_grad = False


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

    def load_state_dict(self, state_dict, strict=True):
        old_state_dict = self.state_dict()
        # assert len(state_dict.keys()) == len(old_state_dict.keys()), (len(state_dict.keys()), len(old_state_dict.keys()))
        if 'features.0.weight' in state_dict:  # 把vgg_net的参数映射成
            self_trainable_param_names = [k for k in self.state_dict().keys() if k.endswith(("weight", "bias", "running_mean", "running_var"))]
            name_map = {my_pname: load_pname for load_pname, my_pname in
                        zip(state_dict.keys(), self_trainable_param_names) if load_pname.startswith("features")}
            new_state_dict = dict()
            for k in self.state_dict().keys():
                if k in name_map:
                    new_state_dict[k] = state_dict[name_map[k]]
                else:
                    new_state_dict[k] = old_state_dict[k]
            super(self.__class__, self).load_state_dict(new_state_dict)  # 只改变parameter的值，不改变requires_grad
        else:
            super(self.__class__, self).load_state_dict(state_dict)
