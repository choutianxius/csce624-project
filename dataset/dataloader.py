import os
import torch
import numpy as np


class CnnDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_base_dir, split, categories, transform=None, target_transform=None
    ):
        self.original_bitmaps = []
        self.masked_bitmaps = []
        self.labels = []
        for category in categories:
            original_path = os.path.join(
                data_base_dir, "sampled_bitmap", split, f"{category}.npz"
            )
            masked_path = os.path.join(
                data_base_dir, "sampled_masked_bitmap", split, f"{category}.npz"
            )
            with np.load(original_path) as f:
                original_bitmaps = f["bitmaps"]
            self.original_bitmaps.extend(original_bitmaps)
            with np.load(masked_path) as f:
                masked_bitmaps = f["bitmaps"]
            self.masked_bitmaps.extend(masked_bitmaps)
            assert len(original_bitmaps) == len(masked_bitmaps)
            self.labels.extend([category for _ in range(len(original_bitmaps))])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        original_bitmap, masked_bitmap, label = (
            self.original_bitmaps[idx],
            self.masked_bitmaps[idx],
            self.labels[idx],
        )
        if self.transform:
            original_bitmap = self.transform(original_bitmap)
            masked_bitmap = self.transform(masked_bitmap)
        if self.target_transform:
            label = self.target_transform(label)
        return original_bitmap, masked_bitmap, label
