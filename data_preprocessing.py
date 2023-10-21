# Taken from this paper: https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
import numpy as np
from os.path import isfile
from threading import Thread
import gc
from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import savez_compressed
from os import listdir


def load_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    pixels = asarray(image)
    return pixels


def load_faces(directory, n_faces):
    faces = list()
    for filename in listdir(directory):
        pixels = load_image(directory + filename)
        faces.append(pixels)
        if len(faces) >= n_faces:
            break
    return asarray(faces)


def plot_faces(faces, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(faces[i])
    pyplot.show()


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


def run_faces(directory, split, faces, completed_files):
    model = MTCNN()
    for file in split:
        image = directory + file[0]
        pixels = load_image(image)
        face = None

        try:
            face = extract_face(model, pixels)
        except:
            print(f"WARNING: could not extract face from '{image}'")

        if face is None:
            continue

        faces.append(face)
        completed_files.append(file)


def generate_csv_filenames(directory):
    if isfile("filenames.txt"):
        return
    filenames = listdir(directory)
    with open("filenames.txt", "w") as f:
        f.write("\n".join(filenames))


def load_faces_mtcnn(directory):
    checkpoint_count = 0
    completed_files = []
    faces = []
    filenames = []
    thread_count = 0
    num_files_per_thread = 1000
    num_threads = 4

    try:
        with open("filenames.txt", "r") as f:
            filenames = f.readlines()
            filenames = [file.split() for file in filenames]

        splits = np.array_split(filenames, num_files_per_thread)

        threads = [
            Thread(
                target=run_faces, args=(directory, split, faces, completed_files), daemon=True
            )
            for split in splits
        ]
        while thread_count < len(threads):
            thread_idx = [x for x in range(thread_count, num_threads+thread_count)]
            [threads[t].start() for t in thread_idx]
            [threads[t].join() for t in thread_idx]

            savez_compressed(f"img_align_celeba_{checkpoint_count}.npz", asarray(faces))
            checkpoint_count += 1
            thread_count += num_threads
            del faces
            gc.collect()
            faces = []

    except KeyboardInterrupt:
        print(f"total_files = {len(filenames)}")
        print(f"completed_files = {len(completed_files)}")
        for file in completed_files:
            filenames.remove(file)
        print(f"remaining_files = {len(filenames)}")

        if 'faces' in locals():
            savez_compressed(f"img_align_celeba_{hash(filenames[-1][0])}", asarray(faces))

        with open("filenames.txt", "w") as f:
            f.write("\n".join([f[0] for f in filenames]))


def celeba_preproc():
    directory = "img_align_celeba/"
    generate_csv_filenames(directory)
    load_faces_mtcnn(directory)


if __name__ == "__main__":
    celeba_preproc()
