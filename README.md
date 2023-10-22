# Latent Space Learning
This project involves training models to help us understand the latent space of GANs.

## Getting Started
### Prerequisites
- Ubuntu 22.04+
- python 3.9.14
- 64GB+ RAM
- Preferably a powerful GPU

Run the following:
```bash
python -m pip install -r requirements.txt
```

While that is running, download the [celebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Download the aligned and cropped version, not the in-the-wild version.

Unzip it into this directory into a directory called `img_align_celeba`.

Preprocess your data:
```bash
python data_preprocessing.py
```

TBD
