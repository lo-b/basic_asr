from pathlib import Path

import tensorflow as tf

# Path to the dataset which is located in /data/mini_speech_commands; to pull
# the dataset run `dvc pull`.
DATASET_PATH = Path().parent.resolve() / Path('data/mini_speech_commands/')
AUTOTUNE = tf.data.AUTOTUNE
