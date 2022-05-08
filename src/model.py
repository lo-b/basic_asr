from tensorflow.keras import layers, models
from keras import Sequential

from config import DATASET_PATH
from data import spectrogram_ds
from utils import get_commands


def build_model() -> Sequential:
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape

    num_labels = len(get_commands(DATASET_PATH))

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=spectrogram_ds.map(
        map_func=lambda spec, label: spec))

    return models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
