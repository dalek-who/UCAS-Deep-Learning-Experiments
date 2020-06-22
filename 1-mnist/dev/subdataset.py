from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import torch

data = list(range(20))

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)

dataset = MyDataset(data)

# 创建subset.超出长度的index不会报错，而是被略过
subset = Subset(dataset, [0,1,3,4,5,11,17,200])
for d in subset:
    print(d)  # [0,1,3,4,5,11,17]

# subset与原始dataset共享一样的数据内存
data[5] = 50000
for d in subset:
    print(d)  # [0,1,3,4,50000,11,17]

# subset的subset
subsubset = Subset(subset, [1,2,3,5,7,20])
for d in subsubset:
    print(d)  # [1,3,4,]
