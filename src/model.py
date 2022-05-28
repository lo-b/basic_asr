from tensorflow.keras import layers, models
from keras import Sequential


def build_model(spectrogram_ds, commands) -> Sequential:
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape

    num_labels = len(commands)

    return models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(64, 64),
        # Normalize.
        layers.Conv2D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
