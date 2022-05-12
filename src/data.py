import os
from typing import Iterable

import tensorflow as tf
import yaml

from config import AUTOTUNE, DATASET_PATH
from utils import get_commands

Ratios = Iterable[tuple[float, float, float]]
params = yaml.safe_load(open("params.yaml"))["process"]


def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(equal_length,
                                 frame_length=255,
                                 frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def get_spectrogram_and_label_id(audio, label, commands):
    spectrogram = get_spectrogram(audio)
    label_id = tf.math.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files, commands):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(map_func=get_waveform_and_label,
                             num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        map_func=lambda audio, label: get_spectrogram_and_label_id(
            audio, label, commands),
        num_parallel_calls=AUTOTUNE)
    return output_ds


def get_datasets(ratios: Ratios = (.8, .1, .1)):
    """
    Preprocess and return train, validation (development), test and spectrogram
    sets. In particular this method:
      - reads in labels from directories
      - reads in waveforms and transform them to spectrograms
      - splits dataset into train, val & test

    Spectrogram dataset is equivalent to the train set but it is unbatched and
    not prefetched; returned to adapt normalization layer.


    Args:
    ratios (Ratios): ratios of split, defaults to (.8, .1, .1) meaning 80% of
    the data will be in the training set, 10% in the validation set and 10% in
    the test set.
    """

    filenames = tf.io.gfile.glob(str(DATASET_PATH / "*/*"))
    filenames = tf.random.shuffle(filenames)
    commands = get_commands(DATASET_PATH)

    # ratios should add up to exactly 1
    assert sum(ratios) == 1

    train_len = int(len(filenames) * ratios[0])
    val_len = int(len(filenames) * ratios[1])
    test_len = int(len(filenames) * ratios[2])

    train_files = filenames[:train_len]
    val_files = filenames[train_len:train_len + val_len]
    test_files = filenames[-test_len:]

    # assert that after split all samples are used and subsets count up to the
    # length of the amount of files
    assert (len(train_files) + len(val_files) +
            len(test_files)) == len(filenames)

    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(map_func=get_waveform_and_label,
                               num_parallel_calls=AUTOTUNE)

    spectrogram_ds = waveform_ds.map(
        map_func=lambda audio, label: get_spectrogram_and_label_id(
            audio, label, commands),
        num_parallel_calls=AUTOTUNE)

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files, commands)
    test_ds = preprocess_dataset(test_files, commands)

    train_ds = train_ds.batch(params["batch_size"])
    val_ds = val_ds.batch(params["batch_size"])

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, spectrogram_ds
