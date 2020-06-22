import matplotlib

import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
batch_size = 4
LR = 0.001  # learning rate
DOWNLOAD_MNIST = False  # set to False if you have downloaded

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Mnist digits dataset
train_dataset = torchvision.datasets.MNIST(
    root='../data/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,  # download it if you don't have it
)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

print(type(train_dataset))


def imshow(img):
    # 保存与显示
    fig, ax = plt.subplots()
    npimg = img.numpy()
    plt_img = ax.imshow(np.transpose(npimg, (1, 2, 0)))
    fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
    fig.savefig("img")


dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)  # [50,1,28,28]

imshow(torchvision.utils.make_grid(images))
print(''.join('%5s' % classes[labels[j]] for j in range(4)))