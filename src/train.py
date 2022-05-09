import json

from keras.callbacks import CSVLogger
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from config import DATASET_PATH

from data import test_ds, train_ds, val_ds
from model import build_model
from utils import get_commands, save_matrix_as_csv

csv_logger = CSVLogger('log.csv', append=False, separator=',')
model = build_model()

params = yaml.safe_load(open("params.yaml"))["train"]

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
                        csv_logger
                    ])
# Get the loss from saved log
df = pd.read_csv("log.csv")
loss_df = df[["loss"]]
loss_df.to_csv("loss.csv", index=False)

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

test_acc = sum(y_pred == y_true) / len(y_true)

with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": test_acc}, outfile)
