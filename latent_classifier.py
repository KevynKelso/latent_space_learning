from math import sqrt

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from progressive_growing_of_gans.pretrained_gan import run_model
from utils_data import plot_faces

model_name = "autoencoder_v0.0.1"


def latent_classifier():
    model = Sequential(
        [
            Dense(256, activation="relu", input_dim=512),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def smile_autoencoder():
    # last value is smile classification
    encoder = Sequential(
        [
            Dense(256, activation="relu", input_dim=513),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="encoder",
    )

    decoder = Sequential(
        [
            Dense(64, activation="relu", input_dim=1),
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
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
    autoencoder.compile(optimizer="adam", loss=smile_loss)
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

    return dataset


def train(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train, epochs=10, batch_size=32)
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {accuracy*100}%")
    print(f"Test Loss: {loss}")
    model.save(model_name)


def train_latent_classifier():
    dataset = load_smiling_not_smiling_dataset()
    class_labels = dataset[:, -1]
    features = dataset[:, :-1]
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, class_labels, test_size=0.3
    )

    model = latent_classifier()
    train(model, X_train, Y_train, X_test, Y_test)


def train_autoencoder():
    dataset = load_smiling_not_smiling_dataset()
    X_train, X_test = train_test_split(dataset, test_size=0.3)

    autoencoder = smile_autoencoder()
    autoencoder.fit(
        X_train,
        X_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(X_test, X_test),
    )
    autoencoder.save(model_name)


def make_predictions():
    num_faces = 9
    gan_pickle = "./progressive_growing_of_gans/karras2018iclr-celebahq-1024x1024.pkl"
    model = tf.keras.models.load_model(model_name)

    latent_vector = np.random.randn(num_faces, 512)
    prediction = model.predict(latent_vector)
    images = run_model(gan_pickle, latent_vector)
    plot_faces(images, int(sqrt(num_faces)), prediction)


if __name__ == "__main__":
    train_autoencoder()
