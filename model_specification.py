#!/usr/bin/env python3
import sys
from pathlib import Path

from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    Reshape,
)
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot
from numpy import ones, zeros
from numpy.random import randint, randn

from utils_data import load_real_samples


def define_discriminator(in_shape=(80, 80, 3)):
    model = Sequential()
    num_filters = 128

    model.add(Conv2D(num_filters, (5, 5), padding="same", input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(num_filters, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(num_filters, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(num_filters, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(num_filters, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def define_generator(latent_dim):
    model = Sequential()

    n_nodes = 128 * 5 * 5
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 128)))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (5, 5), activation="tanh", padding="same"))
    return model


def define_gan(g_model, d_model):
    d_model.trainable = False

    model = Sequential()

    model.add(g_model)

    model.add(d_model)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model


def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)

    X = dataset[ix]

    y = ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)

    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)

    X = g_model.predict(x_input)

    y = zeros((n_samples, 1))
    return X, y


def save_plot(examples, epoch, n=10):
    examples = (examples + 1) / 2.0

    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)

        pyplot.axis("off")

        pyplot.imshow(examples[i])

    filename = "outputs/generated_plot_e%03d.png" % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples(dataset, n_samples)

    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    print(">Accuracy real: %.0f%%, fake: %.0f%%" % (acc_real * 100, acc_fake * 100))

    save_plot(x_fake, epoch)

    filename = "outputs/generator_model_%03d.h5" % (epoch + 1)
    g_model.save(filename)

    filename = "outputs/discriminator_model_%03d.h5" % (epoch + 1)
    d_model.save(filename)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    mode_collapse_count = 0
    mode_collapse_threshold = 3
    epoch = 0

    while epoch < n_epochs:
        for batch in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            print(
                ">%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f"
                % (epoch + 1, batch + 1, bat_per_epo, d_loss1, d_loss2, g_loss)
            )
            # mode collapse detection
            if g_loss > 1000 and epoch > 1:
                mode_collapse_count += 1
            else:
                mode_collapse_count = 0

            if mode_collapse_count > mode_collapse_threshold:
                print("Mode collapse detected! Restoring model from previous epoch.")
                mode_collapse_count = 0
                epoch -= 1
                g_model.load_weights(f"./outputs/generator_e{epoch}")
                d_model.load_weights(f"./outputs/discriminator_e{epoch}")
                break
        else:
            g_model.save_weights(f"./outputs/generator_e{epoch}")
            d_model.save_weights(f"./outputs/discriminator_e{epoch}")
            if (epoch + 1) % 10 == 0:
                summarize_performance(epoch, g_model, d_model, dataset, latent_dim)
            epoch += 1


def main(npz_dataset):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    latent_dim = 100
    d_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    dataset = load_real_samples(npz_dataset)
    print(dataset.shape)
    train(g_model, d_model, gan_model, dataset, latent_dim)


if __name__ == "__main__":
    main(sys.argv[1])
