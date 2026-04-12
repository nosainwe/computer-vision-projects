import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv3D,
    ConvLSTM2D,
    Input,
)
from tensorflow.keras.models import Model

# ---------------------------
# Configuration
# ---------------------------
NUM_FRAMES = 16          # keeping this short — enough temporal context without killing memory
IMG_SIZE = (128, 128)    # width, height — 128 is the sweet spot between detail and compute cost
THRESHOLD = 0.075        # tuned by hand — anything above this and the model is clearly struggling to reconstruct

# UCF-Crime dataset clips — using these two as the normal/anomaly pair for quick testing
NORMAL_VIDEO = "data/roadaccidents/RoadAccidents009_x264.mp4"
ANOMALY_VIDEO = "data/assault/Assault009_x264.mp4"


# ---------------------------
# Helper: extract frames
# ---------------------------
def extract_frames(video_path, num_frames, size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            # end of video — might have fewer frames than asked for, handled below
            break
        resized = cv2.resize(frame, size)
        # converting BGR -> RGB because opencv loads BGR by default and the model expects RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    # not a crash, just a heads up — short clips can still work, just noisier
    if len(frames) < num_frames:
        print(f"Warning: only {len(frames)} frames extracted from {video_path}")

    # normalizing to [0, 1] here — sigmoid output expects this range
    return np.array(frames, dtype=np.float32) / 255.0


# ---------------------------
# Build ConvLSTM Autoencoder
# ---------------------------
def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # encoder — compressing spatial + temporal features down
    # return_sequences=True keeps the full temporal axis alive through both layers
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = BatchNormalization()(x)

    # decoder — mirroring the encoder back up to 64 filters before reconstruction
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = BatchNormalization()(x)

    # Conv3D collapses back to 3 channels (RGB) — sigmoid keeps output in [0, 1] to match normalized input
    outputs = Conv3D(filters=3, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)

    model = Model(inputs, outputs)
    # mse loss is key here — anomalies = high reconstruction error, that's the whole idea
    model.compile(optimizer='adam', loss='mse')
    return model


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    print("Extracting normal video frames...")
    normal_frames = extract_frames(NORMAL_VIDEO, NUM_FRAMES, IMG_SIZE)
    if normal_frames.shape[0] < NUM_FRAMES:
        print("Insufficient frames. Exiting.")
        return

    # adding batch dim — model expects (batch, T, H, W, C), so (1, 16, 128, 128, 3)
    normal_batch = np.expand_dims(normal_frames, axis=0)

    print("Building and training autoencoder...")
    # input shape excludes the batch dim — (T, H, W, C)
    model = build_autoencoder((NUM_FRAMES, IMG_SIZE[1], IMG_SIZE[0], 3))

    # training on normal video only — the whole point is the model learns what "normal" looks like
    # verbose=0 keeps it quiet, 50 epochs is enough for this clip length
    model.fit(normal_batch, normal_batch, epochs=50, batch_size=1, verbose=0)
    print("Training completed.")

    # baseline check — if mse on normal is already high, threshold needs adjusting
    reconstructed_normal = model.predict(normal_batch)[0]
    mse_normal = np.mean((normal_frames - reconstructed_normal) ** 2, axis=(1, 2, 3))
    print(f"Mean MSE on normal video: {np.mean(mse_normal):.5f}")

    print("Testing on anomalous video...")
    anomaly_frames = extract_frames(ANOMALY_VIDEO, NUM_FRAMES, IMG_SIZE)
    if anomaly_frames.shape[0] < NUM_FRAMES:
        print("Insufficient frames in anomaly video.")
        return

    anomaly_batch = np.expand_dims(anomaly_frames, axis=0)
    reconstructed_anomaly = model.predict(anomaly_batch)[0]

    # per-frame mse — averaging over H, W, C so we get one scalar per frame
    mse_anomaly = np.mean((anomaly_frames - reconstructed_anomaly) ** 2, axis=(1, 2, 3))

    # boolean mask — frames where reconstruction error exceeds threshold
    anomalies = mse_anomaly > THRESHOLD

    # plotting mse per frame with threshold line and shaded anomaly regions
    plt.figure(figsize=(10, 4))
    plt.plot(mse_anomaly, marker='o', color='orange', label='MSE')
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Threshold={THRESHOLD}')

    # fill_between only shades where anomalies=True — makes the flagged frames obvious at a glance
    plt.fill_between(range(len(mse_anomaly)), mse_anomaly, THRESHOLD,
                     where=anomalies, color='red', alpha=0.3, label='Anomaly')

    plt.title("Anomaly Detection: Per-Frame Reconstruction Error")
    plt.xlabel("Frame Index")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # saving before show() — show() can block or clear the figure depending on the backend
    plt.savefig("anomaly_detection_plot.png")
    plt.show()
    print("Plot saved as anomaly_detection_plot.png")


if __name__ == "__main__":
    main()
