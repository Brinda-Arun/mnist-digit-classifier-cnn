# MNIST Handwritten Digit Classification using CNN

A PyTorch implementation of a Convolutional Neural Network for classifying handwritten digits from the MNIST dataset, achieving **97% test accuracy** across all 10 digit classes.

---

## Overview

Handwritten digit recognition is a foundational problem in computer vision. This project implements a two-layer CNN trained end-to-end on the MNIST benchmark dataset. The model learns hierarchical spatial features through convolutional and pooling layers, followed by fully connected layers for classification.

---

## Model Architecture

| Layer | Type | Configuration | Output Shape |
|---|---|---|---|
| Input | — | Grayscale image | 1 × 28 × 28 |
| Conv1 | Conv2D + ReLU | 32 filters, 3×3, stride=1, padding=1 | 32 × 28 × 28 |
| Pool1 | MaxPool2D | 2×2, stride=2 | 32 × 14 × 14 |
| Conv2 | Conv2D + ReLU | 64 filters, 3×3, stride=1, padding=1 | 64 × 14 × 14 |
| Pool2 | MaxPool2D | 2×2, stride=2 | 64 × 7 × 7 |
| Flatten | — | — | 3136 |
| FC1 | Linear + ReLU | 3136 → 128 | 128 |
| FC2 | Linear | 128 → 10 | 10 |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| Epochs | 2 |
| Batch size | 64 |
| Train / Test split | 60,000 / 10,000 |
| Input normalisation | Mean = 0.5, Std = 0.5 |

---

## Results

**Overall test accuracy: 97%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.98 | 0.98 | 0.98 | 980 |
| 1 | 0.99 | 0.99 | 0.99 | 1135 |
| 2 | 0.92 | 0.97 | 0.95 | 1032 |
| 3 | 0.98 | 0.95 | 0.96 | 1010 |
| 4 | 0.98 | 0.97 | 0.97 | 982 |
| 5 | 0.93 | 0.97 | 0.95 | 892 |
| 6 | 0.99 | 0.95 | 0.97 | 958 |
| 7 | 0.97 | 0.95 | 0.96 | 1028 |
| 8 | 0.97 | 0.95 | 0.96 | 974 |
| 9 | 0.95 | 0.96 | 0.96 | 1009 |
| **Weighted avg** | **0.97** | **0.97** | **0.97** | **10000** |

---

## Project Structure

```
mnist-digit-classifier-cnn/
├── mnist_digit_classifier.py   # Model definition, training loop, evaluation
└── README.md
```

---

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
pandas
scikit-learn
```

Install dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn
```

---

## Dataset Setup

Download the MNIST dataset in CSV format from [https://pjreddie.com/projects/mnist-in-csv/](https://pjreddie.com/projects/mnist-in-csv/) and place both files in the project root:

```
mnist_train.csv   # 60,000 samples
mnist_test.csv    # 10,000 samples
```

---

## Usage

```bash
python mnist_digit_classifier.py
```

Training progress is printed every 100 steps. After training, the script outputs test accuracy, confusion matrix, and a full classification report.

---

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat-square)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square)

---

## Academic Context

Computer Vision — M.Eng. Information Technology
SRH Hochschule Heidelberg, Germany
Supervised by Prof. Dr. Milan Gnjatovic
