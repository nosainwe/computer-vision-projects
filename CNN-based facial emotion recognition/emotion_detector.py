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
import cv2

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

        print(f"[Epoch {epoch+1:02d}] Train Acc: {tr_a:.4f} | Val Acc: {va_a:.4f}")

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
# GRAD-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x):
        self.model.zero_grad()
        output = self.model(x)
        class_idx = output.argmax(dim=1)

        loss = output[:, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


def show_gradcam(data_dir: str):
    if not os.path.exists(CHECKPOINT):
        print("Train model first.")
        return

    _, val_loader, dataset = get_loaders(data_dir)

    model = FER_CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    gradcam = GradCAM(model, model.features[-5])

    x, y = next(iter(val_loader))
    x = x.to(device)

    cam = gradcam.generate(x[0].unsqueeze(0))

    img = x[0].cpu().squeeze().numpy()
    cam = cv2.resize(cam, (48, 48))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img = np.uint8((img * 0.5 + 0.5) * 255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay[:, :, ::-1])
    plt.title(f"Grad-CAM: {dataset.classes[y[0]]}")
    plt.axis("off")
    plt.show()


# =============================================================================
# WEBCAM
# =============================================================================

def webcam_inference(data_dir: str):
    if not os.path.exists(CHECKPOINT):
        print("Train model first.")
        return

    _, _, dataset = get_loaders(data_dir)

    model = FER_CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            face = (face / 255.0 - 0.5) / 0.5

            face_tensor = torch.tensor(face, dtype=torch.float32)\
                .unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(face_tensor).argmax(1).item()

            label = dataset.classes[pred]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "evaluate", "gradcam", "webcam"]
    )
    parser.add_argument("--data_dir", default=DATA_DIR)

    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir)

    elif args.mode == "evaluate":
        evaluate(args.data_dir)

    elif args.mode == "gradcam":
        show_gradcam(args.data_dir)

    elif args.mode == "webcam":
        webcam_inference(args.data_dir)
