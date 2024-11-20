# Sketch Recognition Project

## Overview

Transformer-based language models are known to be good at predicting missing data even with incomplete context. Some language models have been introduced in the sketch recognition field, but there is no direct study yet investigating whether the next-token-prediction ability of language models help them excel with incomplete sketches. This project aims to fill this gap and provide insight on how the attention mechanism of language models contribute to their performance in the sketch recognition field.

## Tasks

- [ ] Dataset: Build dataset of incomplete sketches, likely from [Quick, Draw!](https://quickdraw.withgoogle.com/).
- [ ] Baseline models: Search for and implement baseline sketch recognition models. They might be traditional feature-based models, or conventional CNN vision models.
- [ ] Target models: Search for and implement transformer-based sketch recognition models. For example, Sketch-bert.
- [ ] Experiment: Evaluate and compare performances of baseline models and target models, on the built dataset.

## Deliverables

- [ ] Report
- [ ] Code
- [ ] Slides
- [ ] Video

## Dataset

The starting point of our dataset of incomplete sketches is the "simplified drawing files (`ndjson`)" from the [Quick, Draw!](https://quickdraw.withgoogle.com/) dataset.

The Quick, Draw! dataset contains 345 categories. For this project, 50 randomly selected categories are used. In each category, 10000 sketches are randomly sampled, with 8000 as the training set, 1000 as the validation set and 1000 as the test set.

Each sketch is further randomly masked to make them incomplete. The masking process goes as follows:

- Several continuous segments in the original sketch are chosen to be masked.
- The total length of the masked segments is 20% of the original simplified sketch (by point count).
- If one masked segment happens to be in the middle of one original stroke, the original stroke is broken into fragments by the masking.


Finally, when evaluating different models, the dataset with masked sketches is further processed to fit their input requirements:

- For the feature-based baseline model, the dataset above is directly given.
- For the vision-based models, sketches are first converted into 28x28 bitmap images following approaches in this [link](https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262).
- For the transformer-based models, the ndjson sketches are converted into the "Sketch-3" format, following approaches used by SketchRNN ([link](https://github.com/hardmaru/quickdraw-ndjson-to-npz)).
