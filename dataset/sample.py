"""
Sample 10000 sketches from each of the 50 categories and split into training,
validation and test sets.
"""

import random
import os
import pathlib
from tqdm import tqdm


random.seed(123456)

with open(os.path.join(pathlib.Path(__file__).parent, "categories_50.txt"), "r") as f:
    categories_50 = [line.strip().split(",")[0] for line in f.readlines() if line]


def reservoir_sample_lines(f_path: str, k: int) -> list[str]:
    reservoir = []

    with open(f_path, "r") as f:
        for i, line in tqdm(
            enumerate(f), desc=f"Sampling lines from {f_path}", position=1, leave=False
        ):
            if i < k:
                reservoir.append(line.strip())
            else:
                j = int(random.random() * i)
                if j < k:
                    reservoir[j] = line.strip()

    return reservoir


def save_sampled(save_path: str, lines: list[str]):
    with open(save_path, "w") as f:
        for line in lines:
            f.write(line + "\n")


for name in ("training", "validation", "test"):
    save_dir = os.path.join(
        pathlib.Path(__file__).parent.parent,
        "data",
        "sampled",
        name,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

for category in tqdm(categories_50, position=0):
    original_path = os.path.join(
        pathlib.Path(__file__).parent.parent,
        "data",
        "original_simplified",
        f"{category}.ndjson",
    )
    if not os.path.exists(original_path):
        raise ValueError(f"Original ndjson file not found: {original_path}")

    sampled = reservoir_sample_lines(original_path, 10000)
    training = sampled[:8000]
    validation = sampled[8000:9000]
    test = sampled[9000:]

    for name, lines in (
        ("training", training),
        ("validation", validation),
        ("test", test),
    ):
        save_sampled(
            os.path.join(
                pathlib.Path(__file__).parent.parent,
                "data",
                "sampled",
                name,
                f"{category}.ndjson",
            ),
            lines,
        )
