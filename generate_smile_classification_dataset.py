import sys
from os.path import isfile

import cv2
import numpy as np
from PIL import Image

from progressive_growing_of_gans.pretrained_gan import run_model
from utils_data import load_real_samples, plot_faces


def main(gan_pickle):
    smiling_latents = []
    not_smiling_latents = []

    while len(smiling_latents) < 100000:
        if isfile("smiling.npz"):
            smiling_latents = np.load("smiling.npz")["arr_0"]
        if isfile("not_smiling.npz"):
            not_smiling_latents = np.load("not_smiling.npz")["arr_0"]

        print(len(smiling_latents))
        print(smiling_latents.shape)

        num_images = 10
        # Generate latent vectors.
        latents = np.random.randn(num_images, 512)
        images = run_model(gan_pickle, latents)

        smile_cascade = cv2.CascadeClassifier("./gan_project/haarcascade_smile.xml")

        for latent, img in zip(latents, images):
            gray_frame = np.array(Image.fromarray(np.uint8(img)).convert("L"))
            # increasing minNeighbors will decrease false positives only detecting 'wide' smiles
            smiles = smile_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.9, minNeighbors=50
            )
            if len(smiles) > 0:
                latent = np.array([latent])
                smiling_latents = np.append(smiling_latents, latent, axis=0)
            else:
                latent = np.array([latent])
                not_smiling_latents = np.append(not_smiling_latents, latent, axis=0)

        if len(smiling_latents) and len(not_smiling_latents):
            np.savez_compressed("smiling.npz", smiling_latents)
            np.savez_compressed("not_smiling.npz", not_smiling_latents)


def load_smiling(gan_pickle):
    smiling_latents = np.load("smiling.npz")["arr_0"]
    print(smiling_latents)

    for latent in smiling_latents:
        latent = np.array([latent])
        print(latent)
        print(latent.shape)
        images = run_model(gan_pickle, latent)
        plot_faces(images, 1)


def average_smiling(gan_pickle):
    smiling_latents = np.load("not_smiling.npz")["arr_0"]
    print(smiling_latents.shape)

    avg_latent = smiling_latents.mean(axis=0)
    avg_latent = np.array([avg_latent])
    images = run_model(gan_pickle, avg_latent)
    plot_faces(images, 1)


if __name__ == "__main__":
    average_smiling(sys.argv[1])
