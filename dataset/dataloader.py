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


label2id = {
    "hospital": 0,
    "barn": 1,
    "dishwasher": 2,
    "airplane": 3,
    "wine bottle": 4,
    "bracelet": 5,
    "bee": 6,
    "hand": 7,
    "bandage": 8,
    "cannon": 9,
    "flamingo": 10,
    "t-shirt": 11,
    "saw": 12,
    "dragon": 13,
    "backpack": 14,
    "snorkel": 15,
    "cello": 16,
    "mouth": 17,
    "vase": 18,
    "motorbike": 19,
    "bathtub": 20,
    "dolphin": 21,
    "bird": 22,
    "pizza": 23,
    "shark": 24,
    "police car": 25,
    "potato": 26,
    "sheep": 27,
    "couch": 28,
    "snake": 29,
    "stitches": 30,
    "beach": 31,
    "camel": 32,
    "butterfly": 33,
    "yoga": 34,
    "bucket": 35,
    "brain": 36,
    "line": 37,
    "sun": 38,
    "hourglass": 39,
    "pliers": 40,
    "sink": 41,
    "tornado": 42,
    "leaf": 43,
    "garden hose": 44,
    "frog": 45,
    "camouflage": 46,
    "hat": 47,
    "hammer": 48,
    "campfire": 49,
}

id2label = {
    0: "hospital",
    1: "barn",
    2: "dishwasher",
    3: "airplane",
    4: "wine bottle",
    5: "bracelet",
    6: "bee",
    7: "hand",
    8: "bandage",
    9: "cannon",
    10: "flamingo",
    11: "t-shirt",
    12: "saw",
    13: "dragon",
    14: "backpack",
    15: "snorkel",
    16: "cello",
    17: "mouth",
    18: "vase",
    19: "motorbike",
    20: "bathtub",
    21: "dolphin",
    22: "bird",
    23: "pizza",
    24: "shark",
    25: "police car",
    26: "potato",
    27: "sheep",
    28: "couch",
    29: "snake",
    30: "stitches",
    31: "beach",
    32: "camel",
    33: "butterfly",
    34: "yoga",
    35: "bucket",
    36: "brain",
    37: "line",
    38: "sun",
    39: "hourglass",
    40: "pliers",
    41: "sink",
    42: "tornado",
    43: "leaf",
    44: "garden hose",
    45: "frog",
    46: "camouflage",
    47: "hat",
    48: "hammer",
    49: "campfire",
}
