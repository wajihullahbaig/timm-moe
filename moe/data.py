"""CIFAR-10 held entirely on the GPU: batching and augmentation are tensor ops, so no dataloader workers."""
import math

import torch
import torchvision

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def random_crop(x, pad=4):
    """Per-sample random crop of `x` after zero padding by `pad` pixels."""
    b, _, h, w = x.shape
    padded = torch.nn.functional.pad(x, (pad,) * 4).permute(0, 2, 3, 1)
    dy = torch.randint(0, 2 * pad + 1, (b, 1), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (b, 1), device=x.device)
    rows = dy + torch.arange(h, device=x.device)
    cols = dx + torch.arange(w, device=x.device)
    batch = torch.arange(b, device=x.device)[:, None, None]
    return padded[batch, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


def random_flip(x):
    flip = torch.rand(x.size(0), 1, 1, 1, device=x.device) < 0.5
    return torch.where(flip, x.flip(3), x)


def random_erase(x, p=0.1, size=8):
    """Zero a `size` x `size` square in a random fraction `p` of the samples."""
    b, _, h, w = x.shape
    cy = torch.randint(0, h, (b, 1, 1), device=x.device)
    cx = torch.randint(0, w, (b, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    inside = ((ys - cy).abs() <= size // 2) & ((xs - cx).abs() <= size // 2)
    chosen = torch.rand(b, 1, 1, device=x.device) < p
    return x.masked_fill((inside & chosen).unsqueeze(1), 0.0)


def augment(x):
    return random_erase(random_flip(random_crop(x)))


class GpuLoader:
    """Iterates (images, labels) batches from tensors already on the device."""

    def __init__(self, images, labels, batch_size, train):
        self.images, self.labels = images, labels
        self.batch_size, self.train = batch_size, train

    def __len__(self):
        n = len(self.labels)
        return n // self.batch_size if self.train else math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n, device=self.labels.device) if self.train else torch.arange(n, device=self.labels.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.images[idx]
            yield (augment(x) if self.train else x), self.labels[idx]


def _load_split(root, train, device):
    dataset = torchvision.datasets.CIFAR10(root, train=train, download=True)
    images = torch.from_numpy(dataset.data).to(device).permute(0, 3, 1, 2).float().div(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std, torch.tensor(dataset.targets, device=device)


def cifar10_loaders(root, batch_size, device, val_size=5000, split_seed=0):
    """(train, val, test) loaders; `val_size` training images are held out, with a fixed split."""
    images, labels = _load_split(root, True, device)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(split_seed)).to(device)
    val_idx, train_idx = order[:val_size], order[val_size:]
    train = GpuLoader(images[train_idx], labels[train_idx], batch_size, train=True)
    val = GpuLoader(images[val_idx], labels[val_idx], batch_size, train=False)
    test = GpuLoader(*_load_split(root, False, device), batch_size, train=False)
    return train, val, test
