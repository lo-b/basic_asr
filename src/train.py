import json

import numpy as np
import tensorflow as tf
import yaml

from config import DATASET_PATH
from data import get_datasets
from model import build_model
from utils import get_commands, save_matrix_as_csv, save_plots_as_csv
from custom_metrics import f1_m, precision_m, recall_m
from sklearn.metrics import classification_report

params = yaml.safe_load(open("params.yaml"))["train"]
train_ds, val_ds, test_ds, spectrogram_ds = get_datasets()
commands = get_commands(DATASET_PATH)
model = build_model(spectrogram_ds, commands)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', f1_m, precision_m, recall_m])

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

metrics = history.history
save_matrix_as_csv(y_true, y_pred, commands)
save_plots_as_csv(metrics)

test_acc = sum(y_pred == y_true) / len(y_true)
report = classification_report(y_true, y_pred, output_dict=True)

with open("metrics.json", 'w') as outfile:
    json.dump(
        {
            "accuracy": test_acc,
            "f1:": report["weighted avg"]["precision"],
            "recall:": report["weighted avg"]["precision"],
            "precision:": report["weighted avg"]["f1-score"],
        }, outfile)
