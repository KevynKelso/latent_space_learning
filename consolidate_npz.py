import sys
from glob import glob
from os.path import isfile

import numpy as np

from utils_data import plot_faces


def consolidate():
    output_file = "consolidated.npz"
    if isfile(output_file):
        print(f"file '{output_file}' already exists")
        return

    all_faces = None
    npz_files = glob(f"smiling*.npz", recursive=False)
    for file in npz_files:
        data = np.load(file)
        print(data["arr_0"].shape)
        if not data["arr_0"].shape[0]:
            continue

        if all_faces is not None:
            all_faces = np.concatenate((all_faces, data["arr_0"]))
        else:
            all_faces = data["arr_0"]

    print("Loaded: ", all_faces.shape)
    np.savez_compressed(f"smiling_consolidated.npz", np.asarray(all_faces))


def main():
    consolidate()


if __name__ == "__main__":
    main()
