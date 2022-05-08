import json

from keras.callbacks import CSVLogger
import pandas as pd
import tensorflow as tf
import yaml

from data import train_ds, val_ds
from model import build_model

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

with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": df.iloc[-1]["val_accuracy"]}, outfile)
