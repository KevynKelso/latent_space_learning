from matplotlib import pyplot
from numpy import load


def normalize(X):
    return (X - 127.5) / 127.5


def load_real_samples(npz_dataset, normalize=True):
    data = load(npz_dataset)
    X = data["arr_0"]

    X = X.astype("float32")

    if normalize:
        X = (X - 127.5) / 127.5
    return X


def plot_faces(faces, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(faces[i])
    pyplot.show()
