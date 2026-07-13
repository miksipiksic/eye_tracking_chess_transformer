"""
dataset.py

Format:
  eye_dataset.npz
    arr_0: (261, 256, 256, 3)  — eye images, RGB, float32 [0-255]
    arr_1: (252, 256, 256, 3)  — images with green dot (labels)

  annotation.csv  — green dot extraction
    radius, x, y, image
    10.5, 125.35, 102.05, 0
    ...

"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

IMG_SIZE_ORIG = 256.0   # original resolution


class EyeDataset(Dataset):

    def __init__(self, npz_path, csv_path, img_size=128, augment=False):
        """
        Parameters:
            npz_path:  eye_dataset.npz
            csv_path:  annotation.csv
            img_size:  goal dim
            augment:   True = agressive
        """
        self.img_size = img_size
        self.augment  = augment

        # Load images from NPZ (arr_0 = original images without the dot)
        npz = np.load(npz_path, allow_pickle=True)
        self.images = npz['arr_0']   # (261, 256, 256, 3), float32

        # Load labels from CSV
        df = pd.read_csv(csv_path)

        # arr_0 ima 261 slika, annotation.csv ima 252 reda
        # Uzimamo samo indekse koji postoje u oba
        valid_idx = df['image'].values.astype(int)
        valid_idx = valid_idx[valid_idx < len(self.images)]

        self.images = self.images[valid_idx]   # (252, 256, 256, 3)
        df = df[df['image'] < len(npz['arr_0'])].reset_index(drop=True)

        # Normalizujemo labele na [0, 1]
        self.labels = np.stack([
            df['x'].values      / IMG_SIZE_ORIG,
            df['y'].values      / IMG_SIZE_ORIG,
            df['radius'].values / IMG_SIZE_ORIG,
        ], axis=1).astype(np.float32)   # (252, 3)

        print(f"Dataset loaded: {len(self)} slika")
        print(f"  Image shape:  {self.images.shape}")
        print(f"  Label shape: {self.labels.shape}")
        print(f"  Label range:  [{self.labels.min():.3f}, {self.labels.max():.3f}]")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # PIL slika iz numpy (RGB)
        img = Image.fromarray(self.images[idx].astype(np.uint8), mode='RGB')
        x_n, y_n, r_n = self.labels[idx]

        if self.augment:
            img, x_n, y_n = self._augment(img, x_n, y_n)

        # Resize + ToTensor + Normalize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])(img)

        label = torch.tensor([x_n, y_n, r_n], dtype=torch.float32)
        return img, label

    def _augment(self, img, x_n, y_n):
        """
        Augmentation - update label coords
          - HorizontalFlip: new_x = 1 - old_x
          - Rotation - i coord around the center


          - Brightness/contrast jitter
          - Blur
        """
        W, H = img.size   # 256, 256

        # 1. Random horizontal rotation (50%)
        if random.random() < 0.5:
            img = TF.hflip(img)
            x_n = 1.0 - x_n   

        # 2. Random rotaion
        if random.random() < 0.5:
            angle = random.uniform(-20, 20)
            img   = TF.rotate(img, angle)
            # rotate around the center (0.5, 0.5)
            cx, cy = 0.5, 0.5
            rad    = -np.radians(angle)
            dx, dy = x_n - cx, y_n - cy
            x_n = cx + dx * np.cos(rad) - dy * np.sin(rad)
            y_n = cy + dx * np.sin(rad) + dy * np.cos(rad)
            # Clamp da ostane u [0,1]
            x_n = float(np.clip(x_n, 0.05, 0.95))
            y_n = float(np.clip(y_n, 0.05, 0.95))

        # 3. Brightness + contrast jitter
        if random.random() < 0.7:
            img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
        if random.random() < 0.7:
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))

        # 4. Gaussian blur 
        if random.random() < 0.3:
            img = img.filter(__import__('PIL').ImageFilter.GaussianBlur(
                radius=random.uniform(0.5, 1.5)))

        return img, x_n, y_n


def create_dataloaders(npz_path, csv_path,
                       img_size=128, batch_size=16,
                       val_split=0.15, test_split=0.10,
                       num_workers=0, seed=42):
    """
    train/val/test DataLoaders.

    batch_size=16 - small dataset
    drop_last=True / train loader bc BatchNorm
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = EyeDataset(npz_path, csv_path, img_size, augment=True)
    val_ds   = EyeDataset(npz_path, csv_path, img_size, augment=False)
    test_ds  = EyeDataset(npz_path, csv_path, img_size, augment=False)

    total   = len(train_ds)
    n_test  = max(1, int(total * test_split))
    n_val   = max(1, int(total * val_split))
    n_train = total - n_val - n_test

    print(f"\nPodela dataseta:")
    print(f"  Train: {n_train:4d}  ({n_train/total*100:.1f}%)")
    print(f"  Val:   {n_val:4d}  ({n_val/total*100:.1f}%)")
    print(f"  Test:  {n_test:4d}  ({n_test/total*100:.1f}%)")

    indices   = torch.randperm(total).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(Subset(test_ds, test_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Test dataset.py")
    print("=" * 50)

    train_loader, val_loader, test_loader = create_dataloaders(
        npz_path='data/eye_dataset.npz',
        csv_path='data/annotation.csv',
        img_size=128,
        batch_size=16
    )

    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape:  {imgs.shape}")
    print(f"Label shape:  {labels.shape}")
    print(f"Pixel range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"Label range:  [{labels.min():.4f}, {labels.max():.4f}]")
    print("\nDataset OK!")