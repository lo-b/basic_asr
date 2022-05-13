import json

import numpy as np
import tensorflow as tf
import yaml

from config import DATASET_PATH, ROOT_DIR
from data import get_datasets
from model import build_model
from utils import get_commands, save_matrix_as_csv, save_plots_as_csv
from custom_metrics import f1_m, precision_m, recall_m
from sklearn.metrics import classification_report

params = yaml.safe_load(open(ROOT_DIR / "params.yaml"))["train"]
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

# Save entire model
model.save('model')

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
    weighted_avg = "weighted avg"
    json.dump(
        {
            "train_accuracy": metrics["accuracy"][-1],
            "val_accuracy": metrics["val_accuracy"][-1],
            "test_accuracy": test_acc,
            "train_loss": metrics["loss"][-1],
            "val_loss": metrics["val_loss"][-1],
            "train_f1": metrics["f1_m"][-1],
            "val_f1": metrics["val_f1_m"][-1],
            "test_f1:": report[weighted_avg]["precision"],
            "train_precision": metrics["precision_m"][-1],
            "val_precision": metrics["val_precision_m"][-1],
            "test_precision:": report[weighted_avg]["f1-score"],
            "train_recall": metrics["recall_m"][-1],
            "val_recall": metrics["val_recall_m"][-1],
            "test_recall:": report[weighted_avg]["precision"],
        }, outfile)
