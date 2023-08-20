#! bin/bash

# look for the dataset in cache first
export HF_DATASETS_OFFLINE=1

conda create -n mlops-training python=3.10.12 pipenv -y
conda activate mlops-training
pipenv install --dev --python=$(conda run which python) --site-packages
pipenv install .
pipenv shell
