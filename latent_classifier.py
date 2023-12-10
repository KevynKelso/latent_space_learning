from math import sqrt
from random import randrange

import imageio
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

from plotlearning import PlotLearning
from progressive_growing_of_gans.pretrained_gan import run_model
from utils_data import plot_faces

ae_model_name = "autoencoder_v0.1.5"
model_name = "latent_classifier_v0.0.4"
max_images = 10  # adjust this based on how much vram you have

gan_pickle = "./progressive_growing_of_gans/karras2018iclr-celebahq-1024x1024.pkl"
gsmile_loss = None


def latent_classifier():
    model = Sequential(
        [
            Dense(1024, activation="relu", input_dim=512),
            Dropout(0.5),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(256, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def smile_autoencoder(classifier):
    # input = [n0, n1, n2, .... n511 <- latent vector, n513 <- smile classification]
    num_latent = 64
    encoder = Sequential(
        [
            Dense(513, activation="leaky_relu", input_dim=513),
            BatchNormalization(),
            Dense(256, activation="leaky_relu"),
            Dense(128, activation="leaky_relu"),
            Dense(num_latent, activation="leaky_relu"),
        ],
        name="encoder",
    )
    encoder.summary()

    decoder = Sequential(
        [
            Dense(128, activation="leaky_relu", input_dim=num_latent),
            Dense(256, activation="leaky_relu"),
            BatchNormalization(),
            Dense(512, activation="sigmoid"),
        ],
        name="decoder",
    )
    decoder.summary()

    def smile_loss(y_true, y_pred):
        # Increasing cosine_loss_weight will make other features more stable
        # at the penalty of de-stabling the smile classification. If this is too
        # large, image will not change.
        cosine_loss_weight = 0.010
        feature_true = y_true[:, :-1]
        feature_pred = y_pred

        class_true = y_true[:, -1]
        class_from_model = classifier(feature_pred)
        bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        bce = bce_fn(class_true, class_from_model)

        cosine_similarity_fn = tf.keras.losses.CosineSimilarity(
            axis=1, reduction=tf.keras.losses.Reduction.SUM
        )
        cosine_similarity = cosine_similarity_fn(feature_true, feature_pred)
        avg = (cosine_similarity * cosine_loss_weight + bce) / 2

        return avg

    global gsmile_loss
    gsmile_loss = smile_loss

    autoencoder = Sequential([encoder, decoder], name="autoencoder")
    autoencoder.compile(optimizer="adam", loss=smile_loss)
    autoencoder.summary()
    return autoencoder


def load_smiling_not_smiling_dataset():
    # return np.load("classifier_dataset.npz")["arr_0"]
    smiling_latents = np.load("smiling_consolidated.npz")["arr_0"]
    is_smiling = np.ones((smiling_latents.shape[0], 1))
    smiling_latents = np.hstack((smiling_latents, is_smiling))

    not_smiling_latents = np.load("not_smiling_consolidated.npz")["arr_0"]
    is_not_smiling = np.zeros((not_smiling_latents.shape[0], 1))
    not_smiling_latents = np.hstack((not_smiling_latents, is_not_smiling))

    dataset = np.vstack((not_smiling_latents, smiling_latents))
    np.random.shuffle(dataset)
    print(f"Loaded {dataset.shape}")

    return dataset


def train(model, X_train, Y_train, X_test, Y_test):

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        Y_train,
        epochs=500,
        batch_size=1024,
        validation_data=(X_test, Y_test),
        callbacks=[early_stop],
    )
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {accuracy*100}%")
    print(f"Test Loss: {loss}")
    model.save(model_name)


def train_latent_classifier():
    dataset = load_smiling_not_smiling_dataset()
    class_labels = dataset[:, -1]
    features = dataset[:, :-1]
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, class_labels, test_size=0.2
    )

    model = latent_classifier()
    train(model, X_train, Y_train, X_test, Y_test)


def train_autoencoder():
    dataset = load_smiling_not_smiling_dataset()
    X_train, X_test = train_test_split(dataset, test_size=0.17)

    classifier = tf.keras.models.load_model(model_name)
    autoencoder = smile_autoencoder(classifier)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    autoencoder.fit(
        X_train,
        X_train,
        epochs=1000,
        batch_size=1200,
        shuffle=True,
        validation_data=(X_test, X_test),
        callbacks=[early_stop, PlotLearning()],
    )
    autoencoder.save(ae_model_name)


def make_predictions():
    num_faces = 9
    model = tf.keras.models.load_model(model_name)

    latent_vector = np.random.randn(num_faces, 512)
    prediction = model.predict(latent_vector)
    images = run_model(gan_pickle, latent_vector)
    plot_faces(images, int(sqrt(num_faces)), prediction)


def make_smiling_faces_classifier():
    num_faces = 9
    model = tf.keras.models.load_model(model_name)
    model.summary()

    random_latents = np.random.randn(1000000, 512)
    prediction = model.predict(random_latents)
    ind = np.argpartition(np.array(prediction).flatten(), -num_faces)[-num_faces:]
    top = random_latents[ind]

    images = run_model(gan_pickle, top)
    plot_faces(images, int(sqrt(num_faces)), prediction[ind])


def make_not_smiling_faces_classifier():
    num_faces = 9
    model = tf.keras.models.load_model(model_name)
    model.summary()

    random_latents = np.random.randn(1000000, 512)
    prediction = model.predict(random_latents)
    bot = np.argpartition(np.array(prediction).flatten(), num_faces)[:num_faces]
    bottom = random_latents[bot]

    images = run_model(gan_pickle, bottom)
    plot_faces(images, int(sqrt(num_faces)), prediction[bot])


def make_smiling_faces_ae():
    num_faces = 10
    interpolation_res = 10

    latent_vectors = np.random.randn(num_faces, 512)
    ae_input = np.zeros(shape=(num_faces * interpolation_res, 513))
    index = 0
    for latent_vector in latent_vectors:
        for smile_fac in np.linspace(0, 1, interpolation_res):
            ae_input[index] = np.append(latent_vector, smile_fac)
            index += 1

    autoencoder = tf.keras.models.load_model(
        ae_model_name, custom_objects={"smile_loss": gsmile_loss}
    )
    interp = autoencoder.predict(ae_input)

    split = np.split(interp, num_faces)
    for i, s in enumerate(split):
        plot_interpolations(s, i, num_faces)

    pyplot.show()


def plot_interpolations(interp, n, num_faces):
    images = run_model(gan_pickle, interp)

    for i, img in enumerate(images):
        pyplot.subplot(num_faces, len(images), i + 1 + (n * len(images)))
        pyplot.axis("off")
        pyplot.imshow(img)


def make_interpolation_gif():
    gif_resolution = 100
    latent_vectors = np.random.randn(1, 512)
    ae_input = np.zeros(shape=(gif_resolution, 513))
    index = 0
    for latent_vector in latent_vectors:
        for smile_fac in np.linspace(0, 1, gif_resolution):
            ae_input[index] = np.append(latent_vector, smile_fac)
            index += 1

    autoencoder = tf.keras.models.load_model(
        ae_model_name, custom_objects={"smile_loss": gsmile_loss}
    )
    interp = autoencoder.predict(ae_input)
    split = np.split(interp, max_images)

    gif_file = f"./outputs_v1.4/{randrange(0, 100000)}"
    with imageio.get_writer(f"{gif_file}.gif", mode="I") as writer:
        for interp in split:
            images = run_model(gan_pickle, interp)
            for i in images:
                writer.append_data(i)

    print(gif_file)


if __name__ == "__main__":
    # print("Training Latent Classifier")
    # train_latent_classifier()
    # make_smiling_faces_classifier()
    # make_not_smiling_faces_classifier()
    # make_predictions()
    print("Training Autoencoder")
    train_autoencoder()
    make_smiling_faces_ae()
    make_interpolation_gif()
