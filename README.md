[![Model training CI](https://github.com/lo-b/basic_asr/actions/workflows/cml.yaml/badge.svg)](https://github.com/lo-b/basic_asr/actions/workflows/cml.yaml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=lo-b_basic_asr&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=lo-b_basic_asr)

# Basic ASR | FHICT AI For Society Minor

This is a basic automatic speech recognition (ASR) model able to recognize 8
commands/words.

## Requirements

If you wish to run experiments, training the model, you will need the following
dependencies.

- [Data Version Control](https://dvc.org/doc/install) (DVC): used for MLOps
  practices
> 💡 Make sure DVC (1) is installed, together with the Google Cloud Storage (GCS) driver
> (2), e.g.:
> 1. `pip install dvc`
> 2. `pip install "dvc[gs]"`

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install): used by DVC
- Jupyter notebook (for manual evaluation)

## Usage

### Running new experiments locally

1. Run `dvc pull` to get the training data and custom data (used for manual
evaluation).
2. Edit source code e.g. the `params.yaml` file or `model.py`.
3. Run `dvc exp run` to run a new experiment.
4. Run `dvc exp diff` to see the difference of your experiment, compared to the
   main branch.


### Manually evaluating model performance

#### With custom data using DVC

If you wish to manually test the model, without your own data,  run `dvc pull
custom_data` to get the data. Start a jupyter notebook server and run the
`./src/manual_evaluation.sync.ipynb` notebook.

#### Own Data
To manually test the model with your own data, load in your own speech which
are `.wav` files with similar specs as the data it was trained on:
```
Format                                   : Wave
File size                                : 31.3 KiB
Duration                                 : 1 s 0 ms
Overall bit rate mode                    : Constant
Overall bit rate                         : 256 kb/s

Audio
Format                                   : PCM
Format settings                          : Little / Signed
Codec ID                                 : 1
Duration                                 : 1 s 0 ms
Bit rate mode                            : Constant
Bit rate                                 : 256 kb/s
Channel(s)                               : 1 channel
Sampling rate                            : 16.0 kHz
Bit depth                                : 16 bits
Stream size                              : 31.2 KiB (100%)
```

Check out the [manual evaluation notebook](./src/manual_evaluation.sync.ipynb)
on how to load in the model and make new predictions.

