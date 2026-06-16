import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False

# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

batch_size = 64
data_root = "./data"
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 数据集类别标签

class CutOut:
    """CutOut数据增强：随机遮挡图像局部区域"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

# 数据增强策略：包括随机裁剪、随机水平翻转、颜色抖动、CutOut遮挡
def data_enhance(aug_type = "original"):
    # 原始数据
    transform_original = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 数据增强方法：随机裁剪、随机水平翻转、颜色抖动、CutOut遮挡
    transform_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        CutOut(n_holes=1, length=16)  # CutOut遮挡
    ])

    transform_dict = {
        "original": transform_original,
        "aug": transform_aug
    }

    return transform_dict[aug_type]


def get_datasets(aug_type = "original"):
    """
    加载数据集
    :param aug_type: 选择增强类型: original不增强、aug：增强
    :return: train/val/test
    """
    select_transform = data_enhance(aug_type)
    train_full = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=select_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=data_enhance("original")  # 测试集不使用增强
    )

    train_len = int(0.8 * len(train_full))
    val_len = len(train_full) - train_len
    train_set, val_set = random_split(train_full, [train_len, val_len])
    return train_set, val_set, test_set

# DataLoader
def get_dataloaders(aug_type="original"):
    train_set, val_set, test_set = get_datasets(aug_type)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# 3. 可视化一个batch
def show_batch():
    loader, _, _ = get_dataloaders(aug_type="original")
    images, labels = next(iter(loader))

    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
    std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1)
    img_grid = torchvision.utils.make_grid(images)
    img_grid = img_grid * std + mean
    img_np = img_grid.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title("CIFAR-10数据集图像")
    plt.savefig("CIFAR_batch.png")
    plt.close()

    label_names = [classes[idx] for idx in labels.numpy()]
    print("可视化batch的标签类别：", label_names)

if __name__ == "__main__":
    show_batch()
    train_set, val_set, test_set = get_datasets("original")
    print(f"训练集:{len(train_set)} 验证集:{len(val_set)} 测试集:{len(test_set)}")