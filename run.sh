#!/bin/bash
set -e

# create virtual environment and install dependencies if they're not there.
if [ ! -d "./.venv"]; then
    python -m venv ./.venv
    source ./.venv/bin/activate
    python -m pip install -r requirements.txt
else
    source ./.venv/bin/activate
fi


# Pre-process celeba dataset to get npz files
dataset = "img_align_celeba"
python celeba_splitter.py
for dir in $(ls $dataset)
do
    python data_preprocessing.py $dir
done
python consolidate_npz.py $dataset

# Train the GAN using the npz file
python model_specification.py consolidated.npz
