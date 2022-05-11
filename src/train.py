import json

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from config import DATASET_PATH
from data import get_datasets
from model import build_model
from utils import get_commands, save_matrix_as_csv, save_plots_as_csv

params = yaml.safe_load(open("params.yaml"))["train"]
train_ds, val_ds, test_ds, spectrogram_ds = get_datasets()
model = build_model(spectrogram_ds, get_commands(DATASET_PATH))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=params["epochs"],
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(verbose=1,
                                                         patience=2),
                    ])

# save only weights of the model since we know the architecture
model.save('model_weights.h5')

test_audio = []
test_labels = []

for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

save_matrix_as_csv(y_true, y_pred, get_commands(DATASET_PATH))
save_plots_as_csv(history.history)

test_acc = sum(y_pred == y_true) / len(y_true)

with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": test_acc}, outfile)
