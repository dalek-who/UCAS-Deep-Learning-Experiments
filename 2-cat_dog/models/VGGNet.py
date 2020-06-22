import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

class VGGNet(nn.Module):  # VGG-16
    def __init__(self, use_batch_norm: bool=True, fixed_cnn_param: bool=False, tags_num: int=2):
        super(self.__class__, self).__init__()
        self.fixed_cnn_param = fixed_cnn_param
        self.cnn = nn.Sequential(OrderedDict([
            ("stage_1", self._cnn_stage(in_channels=3, out_channels=64, block_num=2, use_batch_norm=use_batch_norm),),
            ("stage_2", self._cnn_stage(in_channels=64, out_channels=128, block_num=2, use_batch_norm=use_batch_norm),),
            ("stage_3", self._cnn_stage(in_channels=128, out_channels=256, block_num=3, use_batch_norm=use_batch_norm),),
            ("stage_4", self._cnn_stage(in_channels=256, out_channels=512, block_num=3, use_batch_norm=use_batch_norm),),
            ("stage_5", self._cnn_stage(in_channels=512, out_channels=512, block_num=3, use_batch_norm=use_batch_norm),),
        ]))

        self.fnn = nn.Sequential(
            self._linear_relu_dropout(linear_i=512*7*7, linear_o=4096, dropout=0.2),
            self._linear_relu_dropout(linear_i=4096, linear_o=4096, dropout=0.2),
            nn.Linear(4096, tags_num)
        )

        if self.fixed_cnn_param:
            for name, param in self.named_parameters():
                if name.startswith("cnn"):
                    param.requires_grad = False

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.fnn(x)
        return x

    def _conv_bn_relu_block(self, in_channels: int, out_channels: int, use_batch_norm: bool=True):
        if use_batch_norm:
            layer_list = [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)),
                ("batch_norm", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU(inplace=True))
            ]
        else:
            layer_list = [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)),
                ("relu", nn.ReLU(inplace=True))
            ]
        return nn.Sequential(OrderedDict(layer_list))

    def _cnn_stage(self, in_channels: int, out_channels: int, block_num: int, use_batch_norm: bool=True):
        block_list = [
            ("block_1", self._conv_bn_relu_block(in_channels, out_channels, use_batch_norm)),
            *[(f"block_{i+1}", self._conv_bn_relu_block(out_channels, out_channels, use_batch_norm)) for i in range(1, block_num)],
            ("max_pooling", nn.MaxPool2d(kernel_size=2, stride=2))
        ]
        return nn.Sequential(OrderedDict(block_list))

    def _linear_relu_dropout(self, linear_i, linear_o, dropout):
        return nn.Sequential(
            nn.Linear(linear_i, linear_o),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def load_state_dict(self, state_dict, strict=True):
        old_state_dict = self.state_dict()
        # assert len(state_dict.keys()) == len(old_state_dict.keys()), (len(state_dict.keys()), len(old_state_dict.keys()))
        if 'features.0.weight' in state_dict:  # 把vgg_net的参数映射成
            self_trainable_param_names = [k for k in self.state_dict().keys() if k.endswith(("weight", "bias", "running_mean", "running_var"))]
            name_map = {my_pname: vgg_pname for vgg_pname, my_pname in
                        zip(state_dict.keys(), self_trainable_param_names) if vgg_pname.startswith("features")}
            new_state_dict = dict()
            for k in self.state_dict().keys():
                if k in name_map:
                    new_state_dict[k] = state_dict[name_map[k]]
                else:
                    new_state_dict[k] = old_state_dict[k]
            super(self.__class__, self).load_state_dict(new_state_dict)  # 只改变parameter的值，不改变requires_grad
        else:
            super(self.__class__, self).load_state_dict(state_dict)




if __name__=="__main__":
    #%%
    # import torchvision.models as models
    # torch_vgg = models.vgg16_bn()
    # param_dir = "/data/users/wangyuanzheng/projects/ucas_DL/2-cat_dog/pretrained_parameters/vgg16_bn-6c64b313.pth"
    # vgg16_state_dict = torch.load(param_dir)
    # torch_vgg.load_state_dict(vgg16_state_dict)
    # #%%
    # #%%
    # m = VGGNet()
    # print(m.state_dict()['cnn.stage_1.block_1.conv.weight'][0])
    # print(m.state_dict()['fnn.2.weight'][0])
    # #%%
    # name_map = {my_pname: vgg_pname for vgg_pname, my_pname in zip(torch_vgg.state_dict().keys(), m.state_dict().keys()) if vgg_pname.startswith("features")}
    # old_state_dict = m.state_dict()
    # torch_vgg_state_dict = torch_vgg.state_dict()
    # new_state_dict = dict()
    # for k in m.state_dict().keys():
    #     if k in name_map:
    #         new_state_dict[k] = torch_vgg_state_dict[name_map[k]]
    #     else:
    #         new_state_dict[k] = old_state_dict[k]
    # m.load_state_dict(new_state_dict)
    # print(m.state_dict()['cnn.stage_1.block_1.conv.weight'][0])
    # print(m.state_dict()['fnn.2.weight'][0])
    param_dir = "/data/users/wangyuanzheng/projects/ucas_DL/2-cat_dog/pretrained_parameters/vgg16_bn-6c64b313.pth"
    vgg16_state_dict = torch.load(param_dir)
    m = VGGNet(fixed_cnn_param=True)
    # m = VGGNet()
    m.load_state_dict(vgg16_state_dict)  # load_state_dict不改变param原有的requires_grad
    optim = torch.optim.Adam(m.parameters())
    pass



