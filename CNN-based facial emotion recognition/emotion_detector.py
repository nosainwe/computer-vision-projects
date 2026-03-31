import os
import argparse
import warnings
from collections import Counter
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = "/kaggle/input/fer2013"
CHECKPOINT  = "best_fer_cnn.pt"

BATCH_SIZE  = 64
LR          = 5e-4
EPOCHS      = 60
NUM_CLASSES = 7
SEED        = 42


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


# =============================================================================
# DEVICE
# =============================================================================

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# =============================================================================
# TRANSFORMS
# =============================================================================

train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# =============================================================================
# DATA
# =============================================================================

def get_loaders(data_dir: str):
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"[INFO] Classes: {train_dataset.classes}")
    print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset


def compute_class_weights(dataset):
    targets = [y for _, y in dataset.samples]
    counts = Counter(targets)

    weights = torch.tensor(
        [1.0 / counts[i] for i in range(NUM_CLASSES)],
        dtype=torch.float
    ).to(device)

    return weights


# =============================================================================
# MODEL
# =============================================================================

class FER_CNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =============================================================================
# TRAINING
# =============================================================================

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            if is_train:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss_sum / len(loader), correct / total


def train(data_dir: str):
    train_loader, val_loader, train_dataset = get_loaders(data_dir)

    model = FER_CNN().to(device)
    weights = compute_class_weights(train_dataset)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        tr_l, tr_a = run_epoch(model, train_loader, criterion, optimizer)
        va_l, va_a = run_epoch(model, val_loader, criterion)

        scheduler.step(va_a)

        history["train_acc"].append(tr_a)
        history["val_acc"].append(va_a)
        history["train_loss"].append(tr_l)
        history["val_loss"].append(va_l)

        print(f"[Epoch {epoch+1:02d}] "
              f"Train Acc: {tr_a:.4f} | Val Acc: {va_a:.4f}")

        if va_a > best_val_acc:
            best_val_acc = va_a
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  [✓] Saved best model ({best_val_acc:.4f})")

    plot_curves(**history)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_dir: str):
    if not os.path.exists(CHECKPOINT):
        print("[ERROR] No checkpoint found.")
        return

    _, val_loader, dataset = get_loaders(data_dir)

    model = FER_CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    print("\n", classification_report(y_true, y_pred, target_names=dataset.classes))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=dataset.classes,
                yticklabels=dataset.classes)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


# =============================================================================
# PLOTS
# =============================================================================

def plot_curves(train_acc, val_acc, train_loss, val_loss):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_acc, label="Train")
    axes[0].plot(val_acc, label="Val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(train_loss, label="Train")
    axes[1].plot(val_loss, label="Val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["train", "evaluate"])
    parser.add_argument("--data_dir", default=DATA_DIR)

    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir)
    elif args.mode == "evaluate":
        evaluate(args.data_dir)
