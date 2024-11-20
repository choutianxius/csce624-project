from dataloader import CnnDataset
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt


data_base_dir = os.path.join(pathlib.Path(__file__).parent.parent, "data")
with open(os.path.join(pathlib.Path(__file__).parent, "categories_50.txt"), "r") as f:
    categories_50 = [line.strip().split(",")[0] for line in f.readlines() if line]


def transform(bitmap):
    return np.reshape(bitmap, (28, 28))


dataset = CnnDataset(
    data_base_dir=data_base_dir,
    split="test",
    categories=categories_50,
    transform=transform,
)


def visualize(idx, dataset):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    image, image_masked, label = dataset[idx]
    ax1.imshow(image.reshape(28, 28), cmap="gray")
    ax2.imshow(image_masked.reshape(28, 28), cmap="gray")
    ax1.set_axis_off()
    ax2.set_axis_off()

    plt.suptitle(label)
    plt.show()


visualize(100, dataset)
