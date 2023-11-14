import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from progressive_growing_of_gans.pretrained_gan import run_model
from utils_data import load_real_samples, plot_faces


def main(gan_pickle):
    not_smiling = []
    smiling_imgs = []
    smiling_latents = []
    not_smiling_latents = []

    while len(smiling_imgs) < 1000:
        num_images = 10
        # Generate latent vectors.
        latents = np.random.randn(num_images, 512)
        images = run_model(gan_pickle, latents)

        smile_cascade = cv2.CascadeClassifier("./gan_project/haarcascade_smile.xml")

        for latent, img in zip(latents, images):
            gray_frame = np.array(Image.fromarray(np.uint8(img)).convert("L"))
            # increasing minNeighbors will decrease false positives only detecting 'wide' smiles
            smiles = smile_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.9, minNeighbors=40
            )
            if len(smiles) > 0:
                smiling_imgs.append(np.uint8(img))
                smiling_latents.append(latent)
            else:
                not_smiling.append(np.uint8(img))
                not_smiling_latents.append(latent)

    np.savez_compressed("smiling.npz", smiling_latents)
    np.savez_compressed("not_smiling.npz", not_smiling_latents)

    print(len(smiling_imgs))
    print(len(not_smiling))

    plot_faces(smiling_imgs, 5)
    plot_faces(not_smiling, 5)


def load_smiling(gan_pickle):
    smiling_latents = np.load("smiling.npz")["arr_0"]
    print(smiling_latents)
    print(smiling_latents.shape)

    for latent in smiling_latents:
        latent = np.array([latent])
        print(latent)
        print(latent.shape)
        images = run_model(gan_pickle, latent)
        plot_faces(images, 1)


if __name__ == "__main__":
    main(sys.argv[1])
