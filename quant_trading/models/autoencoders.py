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
from keras.models import Sequential, Model
from tensorflow import keras

from quant_trading.models import output_writer


class AutoEncoder:
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
        # Transform the input
        X_train, X_test = self.transform(X_train, X_test, self.enable)

        # Train the model
        self.model.compile(loss="mse", optimizer="adam")
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        # Evaluate the model
        X_train_pred = self.model.predict(X_train)

        # Plot the results
        self.plot(X_train, X_train_pred, n, self.model_name)

    def bottleneck(self, X_train, X_test, bottleneck_layer):
        standalone_encoder = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[bottleneck_layer].output,
        )

        X_train_features = standalone_encoder.predict(X_train)

        return X_train_features

    def reset_weights(self):
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, "bias_initializer"):
                layer.bias.initializer.run(session=session)

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
        self.model_name = "LSTM_AutoEncoder"
        self.bottleneck_layer = 0
        self.enable = True

        model = Sequential()
        model.add(LSTM(units=encoding_dim, input_shape=(timesteps, input_dim),))
        model.add(Dropout(rate=drop_prob))
        model.add(RepeatVector(n=timesteps))
        model.add(LSTM(units=encoding_dim, return_sequences=True))
        model.add(Dropout(rate=drop_prob))
        model.add(TimeDistributed(Dense(units=input_dim)))

        self.model = model
        self.model.summary()


class BasicAutoEncoder(AutoEncoder):
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=16, drop_prob=0.2):
        self.model_name = "Basic_AutoEncoder"
        self.bottleneck_layer = 0
        self.enable = False

        model = Sequential()
        model.add(Input(shape=(timesteps,)))
        model.add(Dense(encoding_dim, activation="relu"))
        model.add(Dense(timesteps, activation="sigmoid"))

        self.model = model
        self.model.summary()


class DeepAutoEncoder(AutoEncoder):
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=16, drop_prob=0.2):
        self.model_name = "Deep_AutoEncoder"
        self.bottleneck_layer = 2
        self.enable = False

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
