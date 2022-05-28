# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
from IPython.display import Audio
import matplotlib.pyplot as plt
import tensorflow as tf

from custom_metrics import f1_m, precision_m, recall_m
from utils import get_commands
from data import preprocess_dataset

AUTOTUNE = tf.data.AUTOTUNE
commands = get_commands("../data/mini_speech_commands")

# %% [md]
# <h2>Load model</h2>
#
# Load trained model from disk

# %%
trained_model = tf.keras.models.load_model('../model',
                                           custom_objects={
                                               "f1_m": f1_m,
                                               "precision_m": precision_m,
                                               "recall_m": recall_m
                                           })
trained_model.summary()

# %% [md]
#
# <h2>Load in self spoken sample and use model make prediction</h2>

# %%
sample_file = '../custom_data/commands/left/left_Salli.wav'

sample_ds = preprocess_dataset([str(sample_file)], commands)

for spectrogram, label in sample_ds.batch(1):
    prediction = trained_model(spectrogram)
    plt.bar(commands, tf.nn.softmax(prediction[0]))
    plt.title(f"Predictions {commands[label]}")
    plt.show()

Audio(sample_file, rate=16000)
