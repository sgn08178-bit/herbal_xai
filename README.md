# Herbal Seed Classification with XAI

This repository contains the code for our study on herbal seed classification using a deep learning model and explainable AI (XAI) techniques.

## Overview

We trained a ConvNeXt-Small model to classify five herbal seed species and analyzed the model's decision-making process using Integrated Gradients (IG).
The study further investigates the relationship between CNN attention and interpretable geometric features such as color, texture, and frequency components.

## Dataset

* 5 species: ARSE, ARSS, PJNA, PRDA, PRPE
* Total images: 1,124
* Train/Test split: 80/20
* Note: The dataset is not publicly available.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

```bash
python train.py
```

### 2. Integrated Gradients (XAI)

```bash
python ig.py
```

### 3. Occlusion Ablation

```bash
python occlusion.py
```

## Results

* Test Accuracy: **98.67%**
* 5-fold CV Accuracy: **98.89% ± 0.55%**

## Notes

* Paths to datasets are currently defined in the code and may need to be adjusted.
* Large files such as datasets and model weights are excluded from this repository.

## License

MIT License
