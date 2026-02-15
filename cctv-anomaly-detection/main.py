import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D

# ---------------------------
# Configuration
# ---------------------------
NUM_FRAMES = 16
IMG_SIZE = (128, 128)  # width, height
THRESHOLD = 0.075       # anomaly threshold (tune as needed)

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
            break
        resized = cv2.resize(frame, size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if len(frames) < num_frames:
        print(f"Warning: only {len(frames)} frames extracted from {video_path}")
    return np.array(frames, dtype=np.float32) / 255.0

# ---------------------------
# Build ConvLSTM Autoencoder
# ---------------------------
def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    # Encoder
    x = ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same", return_sequences=True, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=32, kernel_size=(3,3), padding="same", return_sequences=True, activation="relu")(x)
    x = BatchNormalization()(x)
    # Decoder
    x = ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same", return_sequences=True, activation="relu")(x)
    x = BatchNormalization()(x)
    outputs = Conv3D(filters=3, kernel_size=(3,3,3), activation="sigmoid", padding="same")(x)
    model = Model(inputs, outputs)
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
    normal_batch = np.expand_dims(normal_frames, axis=0)  # (1, T, H, W, C)

    print("Building and training autoencoder...")
    model = build_autoencoder((NUM_FRAMES, IMG_SIZE[1], IMG_SIZE[0], 3))
    model.fit(normal_batch, normal_batch, epochs=50, batch_size=1, verbose=0)
    print("Training completed.")

    # Test on normal video to get baseline error
    reconstructed_normal = model.predict(normal_batch)[0]
    mse_normal = np.mean((normal_frames - reconstructed_normal) ** 2, axis=(1,2,3))
    print(f"Mean MSE on normal video: {np.mean(mse_normal):.5f}")

    # Test on anomalous video
    print("Testing on anomalous video...")
    anomaly_frames = extract_frames(ANOMALY_VIDEO, NUM_FRAMES, IMG_SIZE)
    if anomaly_frames.shape[0] < NUM_FRAMES:
        print("Insufficient frames in anomaly video.")
        return
    anomaly_batch = np.expand_dims(anomaly_frames, axis=0)
    reconstructed_anomaly = model.predict(anomaly_batch)[0]
    mse_anomaly = np.mean((anomaly_frames - reconstructed_anomaly) ** 2, axis=(1,2,3))

    # Detect anomalies
    anomalies = mse_anomaly > THRESHOLD

    # Plot results
    plt.figure(figsize=(10,4))
    plt.plot(mse_anomaly, marker='o', color='orange', label='MSE')
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Threshold={THRESHOLD}')
    plt.fill_between(range(len(mse_anomaly)), mse_anomaly, THRESHOLD,
                     where=anomalies, color='red', alpha=0.3, label='Anomaly')
    plt.title("Anomaly Detection: Per-Frame Reconstruction Error")
    plt.xlabel("Frame Index")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("anomaly_detection_plot.png")
    plt.show()
    print("Plot saved as anomaly_detection_plot.png")

if __name__ == "__main__":
    main()
