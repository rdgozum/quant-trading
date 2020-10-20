import numpy as np
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

from quant_trading.models import model_utils


class LSTMAutoEncoder:
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=16, drop_prob=0.2):
        model = Sequential()
        model.add(LSTM(units=encoding_dim, input_shape=(timesteps, input_dim),))
        model.add(Dropout(rate=drop_prob))
        model.add(RepeatVector(n=timesteps))
        model.add(LSTM(units=encoding_dim, return_sequences=True))
        model.add(Dropout(rate=drop_prob))
        model.add(TimeDistributed(Dense(units=input_dim)))

        self.model = model

    def reshape(self, X_train, X_test):
        X_train = np.reshape(X_train, X_train.shape + (1,))
        X_test = np.reshape(X_test, X_test.shape + (1,))

        return X_train, X_test

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=128,
        n=5,
        shuffle=False,
    ):
        print(self.model.summary())

        X_train, X_test = self.reshape(X_train, X_test)

        self.model.compile(loss="mse", optimizer="adam")
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        X_train_pred = self.model.predict(X_train)

        model_utils.plot(X_train, X_train_pred, n)


class BasicAutoEncoder:
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=16, drop_prob=0.2):
        model = Sequential()
        model.add(Input(shape=(timesteps,)))
        model.add(Dense(encoding_dim, activation="relu"))
        model.add(Dense(timesteps, activation="sigmoid"))

        self.model = model

    def reshape(self, X_train, X_test):
        return X_train, X_test

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=128,
        n=5,
        shuffle=False,
    ):
        print(self.model.summary())

        X_train, X_test = self.reshape(X_train, X_test)

        self.model.compile(loss="mse", optimizer="adam")
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        X_train_pred = self.model.predict(X_train)

        model_utils.plot(X_train, X_train_pred, n)


class DeepAutoEncoder:
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

    def reshape(self, X_train, X_test):
        return X_train, X_test

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=128,
        n=5,
        shuffle=False,
    ):
        print(self.model.summary())

        X_train, X_test = self.reshape(X_train, X_test)

        self.model.compile(loss="mse", optimizer="adam")
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        X_train_pred = self.model.predict(X_train)

        model_utils.plot(X_train, X_train_pred, n)
