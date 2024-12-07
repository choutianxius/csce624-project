# Sketch Recognition Project

## Overview

Transformer-based language models are known to be good at predicting missing data even with incomplete context. Some language models have been introduced in the sketch recognition field, but there is no direct study yet investigating how they achieve their superior performances. This project aims to mitigate this gap by testing whether transformer-based sketch recognition models perform well even with incomplete sketches.

## Quick Start

We used Python 3.12 for this project. First, create a virtual environment with `venv`:

```shell
# unix-like
python3.12 -m venv .venv

# Windows
py -3.12 -m venv .venv
```

Then, activate the virtual environment with:

```shell
# unix-like
source .venv/bin/activate

# Windows
.\.venv\Scripts\Activate.ps1
```

Install the dependencies with:

```shell
pip install -r requirements.txt
```

Two special notes:

- PyTorch is not included in the `requirements.txt` file due to the complications of CUDA. We use PyTorch v2.5.1, and you should follow the instructions from PyTorch's website for installing versions compatible with your system's CUDA.
- `cairocffi` is required to convert the `ndjson` sketch data into bitmaps, which depends on the `cairo` library. Please following instructions in this [link](https://doc.courtbouillon.org/cairocffi/stable/overview.html) for installing `cairo` on your system. Specifically, for Windows users, we have tested that `cairo` can be installed as a side effect of installing GTK 4 (see this [documentation](https://www.gtk.org/docs/installations/windows/)).


## Dataset

The starting point of our dataset of incomplete sketches is the "simplified drawing files (`ndjson`)" from the [*Quick, Draw!*](https://quickdraw.withgoogle.com/) dataset.

The *Quick, Draw!* dataset contains 345 categories. For this project, 50 randomly selected categories are used. In each category, 10000 sketches are randomly sampled, with 8000 as the training set, 1000 as the validation set and 1000 as the test set.

Each sketch is further randomly masked to make them incomplete. The masking process goes as follows:

- Several continuous segments in the original sketch are chosen to be masked.
- The total length of the masked segments is 20% of the original simplified sketch (by point count).
- If one masked segment happens to be in the middle of one original stroke, the original stroke is broken into fragments by the masking.

Finally, when evaluating different models, the dataset with masked sketches is further processed to fit their input requirements:

- For the feature-based baseline model, the dataset above is directly used.
- For the vision-based models, sketches are first converted into 28x28 bitmap images following approaches in this [link](https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262).
- For the transformer-based models with sequence input, the ndjson sketches should be converted into the "Sketch-3" format, following approaches used by SketchRNN ([link](https://github.com/hardmaru/quickdraw-ndjson-to-npz)). Since we are not covering these sequence-input models, this conversion step is not performed.

### Scripts

The [`dataset`](./dataset/) folder contains scripts for the entire process from downloading the *Quick, Draw!* dataset to masking the sketches to converting sketchese into bitmaps. The [`dataloader.py`](./dataset/dataloader.py) module also provides a `Dataset` class following PyTorch's API for the convenience of working with PyTorch. The processing steps are as follows:

| Step                                    | Script                                             |
| --------------------------------------- | -------------------------------------------------- |
| Sample categories and download          | [`download.py`](./dataset/download.py)             |
| Sample 10000 sketches for each category | [`sample.py`](./dataset/sample.py)                 |
| Randomly mask points from the sketches  | [`mask.py`](./dataset/mask.py)                     |
| Convert `ndjson` sketches into bitmaps  | [`convert_bitmap.py`](./dataset/convert_bitmap.py) |

You can run the scripts in the current project-base folder in the virtual env with:

```shell
python -m dataset.<module name>
```

Ouput files will be stored in the [`data/`](./data/) folder.

## Model Training and Evaluation

We train or fine-tune models on our dataset with either original or masked sketches. We focus on the multi-classification performance degradation between when working with the masked sketches and when working with the original sketches. Metrics including Top-1, -3, -5 accuracy and F1 macro score are used.

### Feature-Based Models

We use `sklearn` to train a series of traditional feature-based classifiers from the `ndjson` format data. We use a subset of Rubine features, and to compute them we first concatenate strokes in the sketches.

### CNN-Based Model

We fine-tune a [MobileNet-V2 based model](https://huggingface.co/JoshuaKelleyDs/quickdraw-MobileNetV2-1.0-finetune) on our datasets.

### Transformer-Based Model

We fine-tune a [MobileViT-V2 based model](https://huggingface.co/JoshuaKelleyDs/quickdraw-MobileVITV2-2.0-Finetune) on our datasets.

### Run for Youself

You can run all the scripts in the [`models`](./models/) folder using module names in the virtual env:

```shell
python -m models.<cv | feature_based_ml>.<module_name>
```

Results like model weight checkpoints will be saved to the [`save`](./save/) folder.
