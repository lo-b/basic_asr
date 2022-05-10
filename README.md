[![Model training CI](https://github.com/lo-b/basic_asr/actions/workflows/cml.yaml/badge.svg)](https://github.com/lo-b/basic_asr/actions/workflows/cml.yaml)

# Basic ASR | FHICT AI For Society Minor

This is a basic automatic speech recognition (ASR) model able to recognize 8
commands/words.

## Requirements

- [Data Version Control](https://dvc.org/doc/install) (DVC): used for MLOps
  practices
> 💡 Make sure DVC (1) is installed, together with the Google Cloud Storage (GCS) driver
> (2), e.g.:
> 1. `pip install dvc`
> 2. `pip install "dvc[gs]"`

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install): used by DVC

## Usage

After cloning the github repo, set your GCP credentials for the DVC remote:

`dvc remote modify --local gcs-storage credentialpath
'<gcp-service-account-key>'`, where `<gcp-service-account-key>` is the location
of your service account key that has acces to the Storage Cloud Bucket where
the training data is stored.
