#%%
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
from pathlib import Path
from collections import namedtuple
from torchvision import transforms
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#%%
# 训练集>=2000张，测试集>=500张
class CatDogDataset(Dataset):
    label_dict = {"dog":0, "cat": 1}
    Example = namedtuple("Example", ["img", "label"])

    def __init__(self, root, transform=None):
        super(CatDogDataset, self).__init__()
        self.root = Path(root)
        self.img_name_list = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, item):
        img_name = self.img_name_list[item]
        label = self.label_dict[img_name.split(".")[0]]
        img = Image.open(self.root / img_name)
        if self.transform is not None:
            img = self.transform(img)
        return self.Example(img, label)


if __name__=="__main__":

    transform = transforms.Compose([
        transforms.Resize((150, 200), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CatDogDataset("../data/train", transform=transform)
    e = dataset[1]
    train, valid, test, unused = random_split(dataset, [2000, 500, 500, len(dataset)-3000])
    all_shapes = np.array([train[i].img.shape for i in range(len(train))])

    mid = np.median(all_shapes, axis=0)  # shape的中位数 [3. , 374. , 448.]
    mean = np.mean(all_shapes, axis=0)  # shape的平均值: [3. , 360.3, 405.6]
    # x边长统计
    x_counter = Counter(all_shapes[:, 1])
    x_size, x_freq = zip(*sorted(x_counter.items()))
    fig, ax = plt.subplots()
    ax.plot(x_size, x_freq)
    fig.savefig("x_freq.png")

    # 长宽比例统计
    scale = np.round(all_shapes[:,1] / all_shapes[:,2], decimals=2)
    scale_counter = Counter(scale)  # 0.75的比例最大，远多于其他比例
    scale_size, scale_freq = zip(*sorted(scale_counter.items()))
    fig, ax = plt.subplots()
    ax.plot(scale_size, scale_freq)
    fig.savefig("scale_freq.png")

    dl = DataLoader(train, batch_size=8)
    batch = iter(dl).next()

