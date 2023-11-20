from math import sqrt

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from progressive_growing_of_gans.pretrained_gan import run_model
from utils_data import plot_faces

ae_model_name = "autoencoder_v0.0.2"
model_name = "latent_classifier_v0.0.2"


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


def smile_autoencoder():
    # last value is smile classification
    num_latent = 256
    encoder = Sequential(
        [
            Dense(1024, activation="relu", input_dim=513),
            Dense(800, activation="relu"),
            Dense(400, activation="relu"),
            Dense(num_latent, activation="relu"),
            Dense(800, activation="relu"),
            Dense(400, activation="relu"),
            Dense(num_latent, activation="sigmoid"),
        ],
        name="encoder",
    )

    decoder = Sequential(
        [
            Dense(400, activation="relu", input_dim=num_latent),
            Dense(800, activation="relu"),
            Dense(513, activation="sigmoid"),
        ],
        name="decoder",
    )

    def smile_loss(y_true, y_pred):
        feature_true = y_true[:, :-1]
        feature_pred = y_pred[:, :-1]

        class_true = y_true[:, -1]
        class_pred = y_pred[:, -1]

        squared_difference = tf.square(feature_true - feature_pred)
        mse = tf.reduce_mean(squared_difference, axis=-1)

        epsilon = 1e-15  # to prevent log(0)
        class_pred = tf.clip_by_value(
            class_pred, epsilon, 1.0 - epsilon
        )  # Clipping predictions to avoid log(0)
        loss = -class_true * tf.math.log(class_pred) - (1 - class_true) * tf.math.log(
            1 - class_pred
        )
        cross_entropy = tf.reduce_mean(loss)

        return mse + cross_entropy

    autoencoder = Sequential([encoder, decoder], name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="cosine_similarity")
    autoencoder.summary()
    return autoencoder


def load_smiling_not_smiling_dataset():
    smiling_latents = np.load("smiling.npz")["arr_0"]
    is_smiling = np.ones((smiling_latents.shape[0], 1))
    smiling_latents = np.hstack((smiling_latents, is_smiling))

    not_smiling_latents = np.load("not_smiling.npz")["arr_0"]
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
    X_train, X_test = train_test_split(dataset, test_size=0.2)

    autoencoder = smile_autoencoder()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    autoencoder.fit(
        X_train,
        X_train,
        epochs=1000,
        batch_size=1024,
        shuffle=True,
        validation_data=(X_test, X_test),
        callbacks=[early_stop],
    )
    autoencoder.save(ae_model_name)


def make_predictions():
    num_faces = 9
    gan_pickle = "./progressive_growing_of_gans/karras2018iclr-celebahq-1024x1024.pkl"
    model = tf.keras.models.load_model(model_name)

    latent_vector = np.random.randn(num_faces, 512)
    prediction = model.predict(latent_vector)
    images = run_model(gan_pickle, latent_vector)
    plot_faces(images, int(sqrt(num_faces)), prediction)


def make_smiling_faces_classifier():
    num_faces = 9
    gan_pickle = "./progressive_growing_of_gans/karras2018iclr-celebahq-1024x1024.pkl"
    model = tf.keras.models.load_model(model_name)

    random_latents = np.random.randn(100000, 512)
    prediction = model.predict(random_latents)
    # ind = np.argpartition(np.array(prediction).flatten(), -num_faces)[-num_faces:]
    bot = np.argpartition(np.array(prediction).flatten(), num_faces)[:num_faces]
    # top = random_latents[ind]
    bottom = random_latents[bot]
    print(f"bottom = {prediction[bot]}")

    # images = run_model(gan_pickle, top)
    # plot_faces(images, int(sqrt(num_faces)), prediction[ind])

    images = run_model(gan_pickle, bottom)
    plot_faces(images, int(sqrt(num_faces)), prediction[bot])


def make_smiling_faces_ae():
    num_faces = 9
    gan_pickle = "./progressive_growing_of_gans/karras2018iclr-celebahq-1024x1024.pkl"
    autoencoder = tf.keras.models.load_model(ae_model_name)

    latent_vectors = np.random.randn(num_faces, 512)

    is_smiling = np.ones((latent_vectors.shape[0], 1))
    not_smiling = np.zeros((latent_vectors.shape[0], 1))

    smiling_latent_vectors = np.hstack((latent_vectors, is_smiling))
    print(smiling_latent_vectors)
    not_smiling_latent_vectors = np.hstack((latent_vectors, not_smiling))

    smiling_latents = autoencoder.predict(smiling_latent_vectors)
    print(smiling_latent_vectors)
    print("\n\n\n")
    print(smiling_latents)
    smiling_latents = smiling_latents[:, :-1]

    print(f"smiling_latents shape = {smiling_latents.shape}")
    print("Smiling Faces :)")

    images = run_model(gan_pickle, smiling_latents)
    plot_faces(images, int(sqrt(num_faces)))

    print(f"not_smiling_latents shape = {not_smiling_latent_vectors.shape}")
    autoencoder = tf.keras.models.load_model(ae_model_name)
    not_smiling_latents = autoencoder.predict(not_smiling_latent_vectors)
    not_smiling_latents = not_smiling_latents[:, :-1]

    print("NOT Smiling Faces :(")
    images = run_model(gan_pickle, not_smiling_latents)
    plot_faces(images, int(sqrt(num_faces)))

    print(f"Interpolation of face 0")
    interpolation = [
        np.hstack((latent_vectors[0], 0)),
        np.hstack((latent_vectors[0], 0.3)),
        np.hstack((latent_vectors[0], 0.4)),
        np.hstack((latent_vectors[0], 0.5)),
        np.hstack((latent_vectors[0], 0.6)),
        np.hstack((latent_vectors[0], 0.7)),
        np.hstack((latent_vectors[0], 0.8)),
        np.hstack((latent_vectors[0], 0.9)),
        np.hstack((latent_vectors[0], 1)),
    ]
    autoencoder = tf.keras.models.load_model(ae_model_name)
    interp_pred = autoencoder.predict(interpolation)[:, :-1]
    images = run_model(gan_pickle, interp_pred)
    plot_faces(images, int(sqrt(num_faces)))


if __name__ == "__main__":
    print("Training Autoencoder")
    train_autoencoder()
    make_smiling_faces_ae()
    print("Training Latent Classifier")
    # train_latent_classifier()
    make_smiling_faces_classifier()
    # make_predictions()
