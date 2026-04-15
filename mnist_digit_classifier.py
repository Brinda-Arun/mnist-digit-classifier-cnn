"""
MNIST Handwritten Digit Classification — Convolutional Neural Network
=====================================================================
Trains a 2-layer CNN on the MNIST dataset using PyTorch.
Achieves 97% test accuracy on 10,000 unseen samples.

Dataset: https://pjreddie.com/projects/mnist-in-csv/
Place mnist_train.csv and mnist_test.csv in the project root before running.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """
    Two-layer Convolutional Neural Network for MNIST digit classification.

    Architecture:
        Conv2D(1→32) + ReLU + MaxPool
        Conv2D(32→64) + ReLU + MaxPool
        Flatten
        Linear(3136→128) + ReLU
        Linear(128→10)
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1   = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2   = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))  # → 32 × 14 × 14
        x = self.maxpool(self.relu(self.conv2(x)))  # → 64 × 7 × 7
        x = self.flatten(x)                          # → 3136
        x = self.relu(self.fc1(x))                  # → 128
        x = self.fc2(x)                              # → 10
        return x


# ---------------------------------------------------------------------------
# Custom dataset
# ---------------------------------------------------------------------------
class CustomMNISTDataset(Dataset):
    """
    Loads MNIST data from CSV files.
    Each row: label, pixel_1, pixel_2, ..., pixel_784
    """

    def __init__(self, csv_file: str, transform=None):
        data             = pd.read_csv(csv_file, header=None)
        self.labels      = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
        self.images      = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(1, 28, 28) / 255.0
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = CustomMNISTDataset('mnist_train.csv', transform=transform)
test_dataset  = CustomMNISTDataset('mnist_test.csv',  transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=0)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model     = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 2

print("\nStarting training...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]  "
                  f"Step [{step + 1:3d}/{len(train_loader)}]  "
                  f"Loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
print("\nEvaluating on test set...\n")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print(f"Test Accuracy : {accuracy_score(all_labels, all_preds):.2%}\n")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))
