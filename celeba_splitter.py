from os import listdir
import numpy as np
from pathlib import Path
from shutil import move
from tqdm import tqdm

directory = "img_align_celeba/"
filenames = []
filenames = listdir(directory)
splits = np.array_split(filenames, 100)
for i, split in tqdm(enumerate(splits)):
    subdirectory = f"img_align_celeba/{i}/"
    Path(subdirectory).mkdir(parents=True, exist_ok=False)
    for file in split:
        move(directory + file, subdirectory + file) 

