#%%
from tempfile import TemporaryFile
import torch
import torch.nn as nn
import json
#%%
class Net(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.linear = nn.Linear(3, 2)
    def forward(self, x):
        return self.linear(x)

#%%
net1: nn.Module = Net()
net2: nn.Module = Net()
with TemporaryFile() as f:
    torch.save(net1.state_dict(), f)
    f.seek(0)
    sd1 = torch.load(f)
    old_sd2 = net2.state_dict()
    print(sd1)
    print(net2.state_dict())
    net2.load_state_dict(sd1)
    print(net2.state_dict())
