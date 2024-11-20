"""
Convert the masked ndjson sketches into flattened 28x28 bitmaps.
"""

import os
import pathlib
import json
from utils import vector_to_raster
import numpy as np
from tqdm import tqdm


with open(os.path.join(pathlib.Path(__file__).parent, "categories_50.txt"), "r") as f:
    categories_50 = [line.strip().split(",")[0] for line in f.readlines() if line]


# masked
for split in ("training", "validation", "test"):
    save_dir = os.path.join(
        pathlib.Path(__file__).parent.parent,
        "data",
        "sampled_masked_bitmap",
        split,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

for category in tqdm(categories_50, position=0):
    for split in ("training", "validation", "test"):
        original_path = os.path.join(
            pathlib.Path(__file__).parent.parent,
            "data",
            "sampled_masked",
            split,
            f"{category}.ndjson",
        )
        if not os.path.exists(original_path):
            raise ValueError(f"Original masked ndjson file not found: {original_path}")

        drawings = []
        with open(original_path, "r") as f:
            for line in f:
                drawing = json.loads(line.strip())["drawing"]
                drawings.append(drawing)
        bitmaps = vector_to_raster(drawings)
        save_path = os.path.join(
            pathlib.Path(__file__).parent.parent,
            "data",
            "sampled_masked_bitmap",
            split,
            f"{category}",
        )
        np.savez_compressed(save_path, bitmaps=bitmaps)


# original
for split in ("training", "validation", "test"):
    save_dir = os.path.join(
        pathlib.Path(__file__).parent.parent,
        "data",
        "sampled_bitmap",
        split,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

for category in tqdm(categories_50, position=0):
    for split in ("training", "validation", "test"):
        original_path = os.path.join(
            pathlib.Path(__file__).parent.parent,
            "data",
            "sampled",
            split,
            f"{category}.ndjson",
        )
        if not os.path.exists(original_path):
            raise ValueError(f"Original ndjson file not found: {original_path}")

        drawings = []
        with open(original_path, "r") as f:
            for line in f:
                drawing = json.loads(line.strip())["drawing"]
                drawings.append(drawing)
        bitmaps = vector_to_raster(drawings)
        save_path = os.path.join(
            pathlib.Path(__file__).parent.parent,
            "data",
            "sampled_bitmap",
            split,
            f"{category}",
        )
        np.savez_compressed(save_path, bitmaps=bitmaps)
