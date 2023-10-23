# Latent Space Learning
This project involves training models to help us understand the latent space of GANs.

## Getting Started
### Prerequisites
- Ubuntu 22.04+
- python 3.9.14
- 64GB+ RAM
- Preferably a powerful GPU

Download the [celebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Download the aligned and cropped version, not the in-the-wild version.

Unzip it into this directory into a directory called `img_align_celeba`.

Preprocess, and train the GAN by running this script.
```bash
./run.sh
```
