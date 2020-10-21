import numpy as np
from abc import ABC, abstractmethod
from keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from keras.models import Sequential
from tensorflow import keras

from quant_trading.models import output_writer


class AutoEncoder(ABC):
    @abstractmethod
    def train(
        self, X_train, y_train, X_test, y_test, epochs, batch_size, n, shuffle,
    ):
        pass

    @staticmethod
    def transform(X_train, X_test, enable=False):
        if enable:
            X_train = np.reshape(X_train, X_train.shape + (1,))
            X_test = np.reshape(X_test, X_test.shape + (1,))

        return X_train, X_test

    @staticmethod
    def plot(X_train, X_train_pred, n, filename):
        output_writer.plot_autoencoder(X_train, X_train_pred, n, filename)


class LSTMAutoEncoder(AutoEncoder):
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=16, drop_prob=0.2):
        model = Sequential()
        model.add(LSTM(units=encoding_dim, input_shape=(timesteps, input_dim),))
        model.add(Dropout(rate=drop_prob))
        model.add(RepeatVector(n=timesteps))
        model.add(LSTM(units=encoding_dim, return_sequences=True))
        model.add(Dropout(rate=drop_prob))
        model.add(TimeDistributed(Dense(units=input_dim)))

        self.model = model
        self.model.summary()

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=128,
        n=None,
        shuffle=False,
    ):
        X_train, X_test = self.transform(X_train, X_test, enable=True)

        self.model.compile(loss="mse", optimizer="adam")
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        X_train_pred = self.model.predict(X_train)

<<<<<<< HEAD
<<<<<<< HEAD
        model_utils.plot(X_train, X_train_pred, n)
=======
        # Plot the results
        self.plot(X_train, X_train_pred, n, filename="lstm_autoencoder")

    def pull_bottleneck(self, X_train, X_test):
        standalone_encoder = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[self.BOTTLENECK_LAYER].output,
        )

        X_train_features = standalone_encoder.predict(X_train)

        return X_train_features
>>>>>>> 6e207a4... Rename
=======
        # Plot the results
        model_utils.plot(X_train, X_train_pred, n, filename="lstm_autoencoder")
>>>>>>> 5ed2667... Save the plot to file


class BasicAutoEncoder(AutoEncoder):
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=16, drop_prob=0.2):
        model = Sequential()
        model.add(Input(shape=(timesteps,)))
        model.add(Dense(encoding_dim, activation="relu"))
        model.add(Dense(timesteps, activation="sigmoid"))

        self.model = model
        self.model.summary()

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=128,
        n=None,
        shuffle=False,
    ):
        X_train, X_test = self.transform(X_train, X_test)

        self.model.compile(loss="mse", optimizer="adam")
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        X_train_pred = self.model.predict(X_train)

<<<<<<< HEAD
<<<<<<< HEAD
        model_utils.plot(X_train, X_train_pred, n)
=======
        # Plot the results
        self.plot(X_train, X_train_pred, n, filename="basic_autoencoder")

    def pull_bottleneck(self, X_train, X_test):
        standalone_encoder = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[self.BOTTLENECK_LAYER].output,
        )

        X_train_features = standalone_encoder.predict(X_train)

        return X_train_features
>>>>>>> 6e207a4... Rename
=======
        # Plot the results
        model_utils.plot(X_train, X_train_pred, n, filename="basic_autoencoder")
>>>>>>> 5ed2667... Save the plot to file


class DeepAutoEncoder(AutoEncoder):
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=16, drop_prob=0.2):
        model = Sequential()
        model.add(Input(shape=(timesteps,)))
        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(encoding_dim, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(timesteps, activation="sigmoid"))

        self.model = model
        self.model.summary()

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=128,
        n=None,
        shuffle=False,
    ):
        X_train, X_test = self.transform(X_train, X_test)

        self.model.compile(loss="mse", optimizer="adam")
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        X_train_pred = self.model.predict(X_train)

<<<<<<< HEAD
<<<<<<< HEAD
        model_utils.plot(X_train, X_train_pred, n)
=======
        # Plot the results
        self.plot(X_train, X_train_pred, n, filename="deep_autoencoder")

    def pull_bottleneck(self, X_train, X_test):
        standalone_encoder = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[self.BOTTLENECK_LAYER].output,
        )

        X_train_features = standalone_encoder.predict(X_train)

        return X_train_features
>>>>>>> 6e207a4... Rename
=======
        # Plot the results
        model_utils.plot(X_train, X_train_pred, n, filename="deep_autoencoder")
>>>>>>> 5ed2667... Save the plot to file
