import json
import random
import math
import os
import pathlib
from tqdm import tqdm


def random_segment_mask(N: int, k: int, L: int) -> set[int]:
    """
    Generate mask indices for a list of length N.
    The masked indices should group to `k` distinct continuous segments,
    and the total length of the masked segments should be L.

    Arguments
    ---------
    N : int
        Total length of the list
    k : int
        Number of distinct segments that the mask indices should group to.
    L : int
        Total length of the grouped mask segments.

    Return
    ------
    A set containing the mask indices.
    """
    if L >= N or L <= 0:
        raise ValueError(f"L ({L}) must be between 0 and N ({N}).")
    if k <= 0 or k > L:
        raise ValueError(f"k ({k}) must be between 1 and L ({L}).")

    lengths = []
    remaining = L
    for _ in range(k - 1):
        length = random.randint(1, remaining - (k - len(lengths) - 1))
        lengths.append(length)
        remaining -= length
    lengths.append(remaining)  # Add the last segment length

    segments = set()
    used_indices = set()

    for length in lengths:
        # Find a valid starting point
        max_start = N - length
        valid_start = False
        while not valid_start:
            start = random.randint(0, max_start)
            if all(idx not in used_indices for idx in range(start, start + length)):
                valid_start = True
                segments.update(range(start, start + length))
                used_indices.update(range(start, start + length))

    return segments


def gen_masked(drawing, ratio=0.2):
    ends = set()
    total_length = 0
    for stroke in drawing:
        total_length += len(stroke[0])
        ends.add(total_length)
    masked_length = math.ceil(total_length * ratio)
    n_masked_segments = random.randint(1, min(masked_length, 2 * len(drawing)))
    masked_indices = random_segment_mask(total_length, n_masked_segments, masked_length)
    i = 0
    masked_drawing = []
    buf_x, buf_y = [], []
    for stroke in drawing:
        for j in range(len(stroke[0])):
            if i in ends or i in masked_indices:
                if buf_x:
                    masked_drawing.append([list(buf_x), list(buf_y)])
                    buf_x.clear()
                    buf_y.clear()
            else:
                buf_x.append(stroke[0][j])
                buf_y.append(stroke[1][j])
            i += 1
    if buf_x:
        masked_drawing.append([list(buf_x), list(buf_y)])
        buf_x.clear()
        buf_y.clear()
    return masked_drawing


if __name__ == "__main__":
    random.seed(123456)

    with open(
        os.path.join(pathlib.Path(__file__).parent, "categories_50.txt"), "r"
    ) as f:
        categories_50 = [line.strip().split(",")[0] for line in f.readlines() if line]

    for split in ("training", "validation", "test"):
        save_dir = os.path.join(
            pathlib.Path(__file__).parent.parent,
            "data",
            "sampled_masked",
            split,
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    for category in tqdm(categories_50, position=0):
        for split in ("training", "validation", "test"):
            raw_path = os.path.join(
                pathlib.Path(__file__).parent.parent,
                "data",
                "sampled",
                split,
                f"{category}.ndjson",
            )
            masked_path = os.path.join(
                pathlib.Path(__file__).parent.parent,
                "data",
                "sampled_masked",
                split,
                f"{category}.ndjson",
            )
            if not os.path.exists(raw_path):
                raise ValueError(f"Original ndjson file not found: {raw_path}")
            with open(raw_path, "r") as f_in:
                with open(masked_path, "w") as f_out:
                    for line in tqdm(f_in, position=1, leave=False):
                        sketch = json.loads(line.strip())
                        sketch_masked = {}
                        sketch_masked["drawing"] = gen_masked(sketch["drawing"])
                        # other fields are omitted
                        f_out.write(
                            json.dumps(sketch_masked, separators=(",", ":")) + "\n"
                        )
