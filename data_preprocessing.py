# Taken from this paper: https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
import sys
from contextlib import redirect_stdout
from os import listdir
from pathlib import Path
from threading import Thread

import numpy as np
from mtcnn.mtcnn import MTCNN
from numpy import asarray, savez_compressed
from PIL import Image
from tqdm import tqdm

checkpoint_count = 0
thread_num = 0


def load_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    pixels = asarray(image)
    image.close()
    return pixels


def load_faces(directory, n_faces):
    faces = list()
    for filename in listdir(directory):
        pixels = load_image(directory + filename)
        faces.append(pixels)
        if len(faces) >= n_faces:
            break
    return asarray(faces)


def thread_func(func):
    def decorator(*args):
        global thread_num
        thread_num += 1
        Path("logs").mkdir(parents=True, exist_ok=True)
        with open(f"logs/thread-{thread_num}.log", "w") as f:
            with redirect_stdout(f):
                func(*args)

    return decorator


def extract_face(model, pixels, required_size=(80, 80)):
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None
    x1, y1, width, height = faces[0]["box"]
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


@thread_func
def run_faces(directory, split):
    model = MTCNN()
    faces = []
    for file in split:
        try:
            image = directory + file
            pixels = load_image(image)
            face = None

            face = extract_face(model, pixels)
            if face is None:
                continue
            faces.append(face)

        except:
            with open("warnings.txt", "a") as f:
                f.write(f"WARNING: could not extract face from '{file}'\n")

    global checkpoint_count
    checkpoint_count += 1
    fname = f"img_align_celeba_{checkpoint_count}.npz"
    savez_compressed(directory + fname, asarray(faces))
    del faces


def grouped(iterable, n):
    return zip(*[iter(iterable)] * n)


def load_faces_mtcnn(directory, num_checkpoints=4, num_threads=4):
    filenames = []
    filenames = listdir(directory)
    for file in filenames:
        if "npz" in file:
            print(f"directory '{directory}' has already been preprocessed")
            return
    splits = np.array_split(filenames, num_checkpoints)

    for split_group in tqdm(grouped(splits, num_threads)):
        threads = [
            Thread(target=run_faces, args=(directory, split), daemon=True)
            for split in split_group
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]


if __name__ == "__main__":
    load_faces_mtcnn(sys.argv[1])
