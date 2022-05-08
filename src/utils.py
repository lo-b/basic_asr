import numpy as np
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
