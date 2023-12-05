# Latent Space Learning
This project involves training models to help us understand the latent space of GANs.

## Getting Started
### Prerequisites
- Ubuntu 22.04+
- python 3.9.14
- 32GB+ RAM
- Preferably a powerful GPU

0. If you haven't already, clone this repo, and run `git submodule init` and `git submodule update`.
The progressive_growing_of_gans submodule is a fork of a repo NVIDIA created. The fork changes are strictly tensorflow 2 and ubuntu 22.04 compatability changes to work with their pre-trained pickle files.

1. Download the [NVIDIA GAN pickle file](https://drive.google.com/file/d/188K19ucknC6wg1R6jbuPEhTq9zoufOx4/view?usp=drive_link)
This is a pre-trained state of the art 2018 GAN to be used in experimentation. NVIDIA was kind enough to publish their pre-trained model on google drive.
Please move the pickle file to the progressive_growing_of_gans directory as there exists some hardcoded paths to it. Please understand this project is a work in progress.

2. Test the NVIDIA GAN is working properly.
`python progressive_growing_of_gans/pretrained_gan.py <path-to-pickle-file>/karras2018iclr-celebahq-1024x1024.pkl`
If it is working, you should see 10 generated images show up in the progressive_growing_of_gans directory.

3. Download our [latent space smiling classification dataset](https://drive.google.com/file/d/1pZ7p2OqQL6hsZkg_cAJ0pIINqwrgpL1V/view?usp=drive_link) and [latent space not smiling classification dataset](https://drive.google.com/file/d/11Mtv9w6mRCbS3q96ks4ntoEQmmvw1JEz/view?usp=drive_link)
Please move them into this directory.

4. Run `python latent_classifier.py`. This should generate some plots and finally a gif with a smile interpolation in a versioned output folder.

### Known Issues
- Calling autoencoder.predict multiple times in the same python instance results in an error due to the custom loss function.
