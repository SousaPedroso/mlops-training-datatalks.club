#! bin/bash

# look for the dataset in cache first
export HF_DATASETS_OFFLINE=1

dataset_dir = $(pwd)/data

conda create -n mlops-training python=3.10 pipenv
conda activate mlops-training
pipenv install .
pipenv shell