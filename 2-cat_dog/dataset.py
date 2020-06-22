from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
from pathlib import Path
from collections import namedtuple
from torchvision import transforms
import numpy as np


# 训练集>=2000张，测试集>=500张
class CatDogDataset(Dataset):
    label_dict = {"dog":0, "cat": 1}
    Example = namedtuple("Example", ["img", "label", "name"])

    def __init__(self, root, transform=None):
        super(CatDogDataset, self).__init__()
        self.root = Path(root)
        self.img_name_list = os.listdir(root)
        self.transform = transform
        self._cache = dict()

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, item):
        if item not in self._cache:
            img_name = self.img_name_list[item]
            label = self.label_dict[img_name.split(".")[0]]
            img = Image.open(self.root / img_name)
            if self.transform is not None:
                img = self.transform(img)
            example = self.Example(img, label, img_name)
            self._cache[item] = example
        else:
            example = self._cache[item]
        return example
