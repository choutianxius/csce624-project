from transformers import AutoImageProcessor, AutoModelForImageClassification
from dataset.dataloader import CnnDataset, label2id
import os
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
import json


parser = argparse.ArgumentParser(
    description="Fine-tune one of the supported computer-vision based sketch-recognition models on the custom dataset."
)
parser.add_argument(
    "--model",
    type=str,
    choices=["mobilenet", "mobilevit"],
    required=True,
    help="The model to investigate",
)
parser.add_argument(
    "--masked",
    action="store_true",
    help="Whether to fine-tune on the masked or unmasked dataset",
)

args = parser.parse_args()
model_name = (
    "JoshuaKelleyDs/quickdraw-MobileNetV2-1.0-finetune"
    if args.model == "mobilenet"
    else "JoshuaKelleyDs/quickdraw-MobileVITV2-2.0-Finetune"
)
print(
    f"Starting from {model_name} on {"masked" if args.masked else "unmasked"} dataset"
)


with open(
    os.path.join(
        pathlib.Path(__file__).parent.parent.parent, "dataset", "categories_50.txt"
    ),
    "r",
) as f:
    categories_50 = [line.strip().split(",")[0] for line in f.readlines() if line]


def transform(bitmap):
    return np.reshape(bitmap, (1, 28, 28))


def target_transform(label):
    return label2id[label]


dataset = CnnDataset(
    data_base_dir=os.path.join(pathlib.Path(__file__).parent.parent.parent, "data"),
    split="training",
    categories=categories_50,
    transform=transform,
    target_transform=target_transform,
)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# adjust out layer
model.classifier = torch.nn.Linear(model.classifier.in_features, 50)

# fine-tuning set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# fine-tuning
num_epochs = 10
losses = []
for epoch in tqdm(range(num_epochs), position=0):
    t0 = time.time()
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, position=1, leave=False):
        original, masked, label_ids = batch
        input_ = processor(masked if args.masked else original, return_tensors="pt")[
            "pixel_values"
        ]
        input_ = input_.to(device)
        target = label_ids.to(device)

        optimizer.zero_grad()
        output = model(pixel_values=input_).logits
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())

    print(
        f"Epoch {epoch}, total time = {time.time() - t0}, average batch loss = {total_loss / len(dataloader)}"
    )

save_dir = os.path.join(
    pathlib.Path(__file__).parent.parent.parent,
    "save",
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch.save(
    model,
    os.path.join(
        save_dir,
        f"{args.model}{"_masked" if args.masked else ""}_finetune.pth",
    ),
)

plt.plot(losses)
plt.savefig(
    os.path.join(
        save_dir,
        f"{args.model}{"_masked" if args.masked else ""}_finetune_loss.png",
    )
)

with open(
    os.path.join(
        save_dir,
        f"{args.model}{"_masked" if args.masked else ""}_finetune_loss.json",
    ),
    "w",
) as loss_json_f:
    json.dump(
        losses,
        loss_json_f,
    )
