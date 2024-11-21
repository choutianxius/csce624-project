from transformers import AutoImageProcessor
from dataset.dataloader import CnnDataset, label2id
import os
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import top_k_accuracy_score, f1_score


parser = argparse.ArgumentParser(
    description="Evaluate one of the supported computer-vision based sketch-recognition models "
    + "which have been fine-tuned on the custom dataset."
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
    f"Evaluating {model_name} fine-tuned on {"masked" if args.masked else "unmasked"} dataset"
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
    split="test",
    categories=categories_50,
    transform=transform,
    target_transform=target_transform,
)
dataloader = DataLoader(dataset=dataset, batch_size=128)


processor = AutoImageProcessor.from_pretrained(model_name)
save_path = os.path.join(
    pathlib.Path(__file__).parent.parent.parent,
    "save",
    f"{args.model}{"_masked" if args.masked else ""}_finetune.pth",
)
if not os.path.exists(save_path):
    raise ValueError(f"Saved model path doesn't exist: {save_path}")
model = torch.load(save_path)
model.eval()

# fine-tuning set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model.to(device)

probs = []
targets = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        original, masked, label_ids = batch
        input_ = processor(masked if args.masked else original, return_tensors="pt")[
            "pixel_values"
        ]
        input_ = input_.to(device)
        target = label_ids.to(device)
        output = model(pixel_values=input_).logits
        prob = torch.softmax(output, dim=-1)
        targets.extend(target.cpu().tolist())
        probs.extend(prob.cpu().tolist())

probs = np.array(probs)
targets = np.array(targets)

# Top-1 accuracy
top_1_acc = top_k_accuracy_score(targets, probs, k=1, labels=np.arange(probs.shape[1]))

# Top-3 accuracy
top_3_acc = top_k_accuracy_score(targets, probs, k=3, labels=np.arange(probs.shape[1]))

# Top-5 accuracy
top_5_acc = top_k_accuracy_score(targets, probs, k=5, labels=np.arange(probs.shape[1]))

# F1-macro score
predicted_labels = np.argmax(probs, axis=1)  # Predicted labels based on max probability
f1_macro = f1_score(targets, predicted_labels, average="macro")

# Print results
print(f"Top-1 Accuracy: {top_1_acc:.4f}")
print(f"Top-3 Accuracy: {top_3_acc:.4f}")
print(f"Top-5 Accuracy: {top_5_acc:.4f}")
print(f"F1-Macro Score: {f1_macro:.4f}")
