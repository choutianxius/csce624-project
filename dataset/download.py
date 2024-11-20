"""
Download 50 randomly selected categories of the Quick, Draw! dataset as .ndjson files.
"""

import random
import os
import pathlib
import requests
from tqdm import tqdm


random.seed(123456)

# read original categories
# see https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt
categories_path = os.path.join(pathlib.Path(__file__).parent, "categories.txt")

with open(categories_path, "r") as f:
    categories = list(filter(bool, [line.strip() for line in f.readlines()]))

# select 50 categories
categories_50 = random.sample(list(enumerate(categories)), k=50)
assert len(categories_50) == len(set(categories_50))

categories_50_path = os.path.join(pathlib.Path(__file__).parent, "categories_50.txt")
with open(categories_50_path, "w") as f:
    for idx, category in categories_50:
        f.write(f"{category},{idx}\n")

# download
data_dir_path = os.path.join(
    pathlib.Path(__file__).parent.parent, "data", "original_simplified"
)
if not os.path.exists(data_dir_path):
    os.makedirs(data_dir_path)
for _, category in tqdm(categories_50):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
    save_path = os.path.join(data_dir_path, f"{category}.ndjson")
    if os.path.exists(save_path):
        continue
    res = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
