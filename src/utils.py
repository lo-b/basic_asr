from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


def get_commands(data_dir):
    """get_commands.

    Get all commands that the ASR model is being trained on. The directory
    where the training data is stored should have subdirectories for each
    respective commands.

    Args:
        data_dir: the directory where audio smaples are stored
    """
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    return commands


def save_matrix_as_csv(actual: 'np.ndarray[Any]', predicted: 'np.ndarray[Any]',
                       commands: list[str]) -> None:
    """save_as_matrix_csv.

    Helper function to save a confusion matrix to be used by CML + DVC in an
    ML-pipeline as a CSV file.

    Args:
        actual: true labels
        predicted: predicted labels

    Returns:
        None:
    """
    # first copy actual and predicted as stringn ndarrays
    actual = actual.copy().astype(str)
    predicted = predicted.copy().astype(str)

    # Replace int labels with actual names in both actual & predicted
    for i in range(len(commands)):
        actual[actual == str(i)] = commands[i]
        predicted[predicted == str(i)] = commands[i]

    df = pd.DataFrame({"actual": actual, "predicted": predicted})

    df.to_csv("matrix.csv", index=False)


def save_matrix_as_png(actual: 'np.ndarray[Any]', predicted: 'np.ndarray[Any]',
                       commands: list[str]):
    confusion_mtx = tf.math.confusion_matrix(actual, predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=commands,
                yticklabels=commands,
                annot=True,
                fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig("out.png")
