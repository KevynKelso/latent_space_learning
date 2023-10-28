import sys
from glob import glob
from os.path import isfile

import numpy as np

from utils_data import plot_faces


def consolidate(img_dir):
    output_file = "consolidated.npz"
    if isfile(output_file):
        print(f"file '{output_file}' already exists")
        return

    all_faces = None
    npz_files = glob(f"{img_dir}/**/*.npz", recursive=True)
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
    plot_faces(all_faces, 5)
    np.savez_compressed(f"consolidated.npz", np.asarray(all_faces))
    import os

    import psutil

    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


def main():
    consolidate(sys.argv[1])


if __name__ == "__main__":
    main()
