# Load the saved best checkpoint and evaluate it on the dataset's test split.
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

    # Report precision, recall and F1-score for each class.
    print("\n", classification_report(y_true, y_pred, target_names=dataset.classes))

    # Rows represent true classes and columns represent predicted classes.
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

# Grad-CAM uses activations and gradients from a convolutional layer to highlight image regions influencing a prediction.
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
        # Weight each feature map by its average gradient, then combine maps into a class-activation map.
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


# Display one test image with its Grad-CAM heatmap overlaid.
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

# Open the webcam, detect faces with a Haar cascade, and classify each detected face in real time.
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
        # Detect face rectangles; 1.3 is the scale factor and 5 is the neighbour threshold.

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            face = (face / 255.0 - 0.5) / 0.5  # Match the training normalisation.

            face_tensor = torch.tensor(face, dtype=torch.float32)\
                .unsqueeze(0).unsqueeze(0).to(device)

            # Inference does not need gradient tracking, so this uses less memory.
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

# Plot and save accuracy and loss curves so training behaviour and possible overfitting can be inspected.
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
