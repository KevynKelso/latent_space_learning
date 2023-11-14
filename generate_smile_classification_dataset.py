import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from progressive_growing_of_gans.pretrained_gan import run_model
from utils_data import load_real_samples, plot_faces


def main(gan_pickle):
    num_images = 50
    # Generate latent vectors.
    latents = np.random.RandomState(901).randn(num_images, 512)
    images = run_model(gan_pickle, latents)

    smile_cascade = cv2.CascadeClassifier("./gan_project/haarcascade_smile.xml")
    not_smiling = []
    smiling_imgs = []

    for img in tqdm(images):
        gray_frame = np.array(Image.fromarray(np.uint8(img)).convert("L"))
        # increasing minNeighbors will decrease false positives only detecting 'wide' smiles
        smiles = smile_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.9, minNeighbors=30
        )
        if len(smiles) > 0:
            smiling_imgs.append(np.uint8(img))
        else:
            not_smiling.append(np.uint8(img))

    print(len(smiling_imgs))
    print(len(not_smiling))

    plot_faces(smiling_imgs, 3)
    plot_faces(not_smiling, 7)


if __name__ == "__main__":
    main(sys.argv[1])
