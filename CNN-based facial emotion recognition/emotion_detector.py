"""
emotion_detector.py
-------------------
CNN-based facial emotion recognition trained on FER-2013.

Classifies 48x48 grayscale face images into 7 emotion categories:
    Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

Dataset: FER-2013
    https://www.kaggle.com/datasets/msambare/fer2013

Usage:
    python emotion_detector.py --mode train
    python emotion_detector.py --mode evaluate
    python emotion_detector.py --mode gallery
"""

import os
import argparse
import warnings
from collections import Counter

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

# change DATA_DIR to wherever you extracted the FER-2013 dataset
DATA_DIR    = "/kaggle/input/fer2013"
CHECKPOINT  = "best_fer_cnn.pt"

BATCH_SIZE  = 64
LR          = 5e-4
EPOCHS      = 60
NUM_CLASSES = 7


# =============================================================================
# DEVICE
# =============================================================================

# cudnn.benchmark lets cudnn auto-tune kernel selection — speeds up fixed-size inputs
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# =============================================================================
# TRANSFORMS
# =============================================================================

# train: flip + rotation augmentation — FER images are tiny (48x48) so keeping it light
train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # normalizing to [-1, 1] for grayscale
])

# val/test: no augmentation — deterministic outputs
val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# =============================================================================
# DATA LOADERS
# =============================================================================

def get_loaders(data_dir: str):
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=val_transform
    )

    # pin_memory=True speeds up CPU->GPU transfers — only helps when using CUDA
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_loader, val_loader, train_dataset


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    # inverse frequency weighting — gives more importance to underrepresented emotions
    # disgust is severely underrepresented in FER-2013, this matters a lot
    targets     = [y for _, y in dataset.samples]
    class_count = Counter(targets)
    weights     = torch.tensor(
        [1.0 / class_count[i] for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    return weights


# =============================================================================
# MODEL
# =============================================================================

class FER_CNN(nn.Module):
    """
    4-block CNN: progressive feature extraction (64 -> 128 -> 256 -> 512 filters)
    Each block: Conv -> ReLU -> BN -> MaxPool -> Dropout
    Classifier head: Linear(512*3*3, 256) -> Linear(256, 512) -> Linear(512, 7)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # feature extractor — doubling channels each block until 512
        # BatchNorm after ReLU stabilizes training at small image sizes like 48x48
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 5, padding=2),   # 5x5 kernel catches broader facial features
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
        # 48x48 -> 24 -> 12 -> 6 -> 3 after 4 MaxPool2d(2) ops — so 512 * 3 * 3 = 4608
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# =============================================================================
# TRAINING
# =============================================================================

def run_epoch(model, loader, criterion, optimizer, train: bool = True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            out  = model(x)
            loss = criterion(out, y)

            if train:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()
            correct  += (out.argmax(1) == y).sum().item()
            total    += y.size(0)

    return loss_sum / len(loader), correct / total


def train(data_dir: str = DATA_DIR):
    train_loader, val_loader, train_dataset = get_loaders(data_dir)

    weights   = compute_class_weights(train_dataset, NUM_CLASSES)
    model     = FER_CNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ReduceLROnPlateau on val accuracy — halves lr after 3 epochs without improvement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        tr_l, tr_a = run_epoch(model, train_loader, criterion, optimizer, train=True)
        va_l, va_a = run_epoch(model, val_loader,   criterion, optimizer, train=False)

        scheduler.step(va_a)

        train_losses.append(tr_l);  train_accs.append(tr_a)
        val_losses.append(va_l);    val_accs.append(va_a)

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
            f"Train Loss: {tr_l:.4f} | Train Acc: {tr_a:.4f} | "
            f"Val Loss: {va_l:.4f} | Val Acc: {va_a:.4f}"
        )

        # saving best model by val accuracy
        if va_a > best_val_acc:
            best_val_acc = va_a
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  Checkpoint saved (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    plot_curves(train_accs, val_accs, train_losses, val_losses)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_dir: str = DATA_DIR):
    if not os.path.exists(CHECKPOINT):
        print(f"No checkpoint found at {CHECKPOINT}. Train first.")
        return

    _, val_loader, train_dataset = get_loaders(data_dir)

    model = FER_CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x   = x.to(device)
            out = model(x)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


# =============================================================================
# GALLERY
# =============================================================================

def prediction_gallery(data_dir: str = DATA_DIR, n: int = 10):
    if not os.path.exists(CHECKPOINT):
        print(f"No checkpoint found at {CHECKPOINT}. Train first.")
        return

    _, val_loader, train_dataset = get_loaders(data_dir)

    model = FER_CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # green title = correct, red = wrong — fast visual scan of where the model fails
    shown = 0
    plt.figure(figsize=(12, 6))

    with torch.no_grad():
        for x, y in val_loader:
            x, y  = x.to(device), y.to(device)
            preds = model(x).argmax(1)

            for i in range(x.size(0)):
                if shown >= n:
                    break
                img  = x[i].cpu().squeeze()
                true = train_dataset.classes[y[i]]
                pred = train_dataset.classes[preds[i]]

                plt.subplot(2, 5, shown + 1)
                plt.imshow(img, cmap="gray")
                plt.title(
                    f"T: {true}\nP: {pred}",
                    color="green" if true == pred else "red",
                    fontsize=9
                )
                plt.axis("off")
                shown += 1

            if shown >= n:
                break

    plt.suptitle("Prediction Gallery", fontsize=13)
    plt.tight_layout()
    plt.savefig("prediction_gallery.png", dpi=150)
    plt.show()
    print("Saved: prediction_gallery.png")


# =============================================================================
# PLOTS
# =============================================================================

def plot_curves(train_accs, val_accs, train_losses, val_losses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # accuracy
    axes[0].plot(train_accs, label="Train", linewidth=2)
    axes[0].plot(val_accs,   label="Val",   linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # loss — watching the train/val gap to catch overfitting early
    axes[1].plot(train_losses, label="Train", linewidth=2)
    axes[1].plot(val_losses,   label="Val",   linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved: training_curves.png")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "evaluate", "gallery"], required=True
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help="Path to FER-2013 dataset root (expects train/ and test/ subdirs)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir)
    elif args.mode == "evaluate":
        evaluate(args.data_dir)
    elif args.mode == "gallery":
        prediction_gallery(args.data_dir)
