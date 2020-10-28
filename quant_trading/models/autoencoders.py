import numpy as np
from keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    LeakyReLU,
)
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import GlorotUniform, Zeros

from quant_trading.models.model_utils import plot_autoencoder


class AutoEncoder:
    def train(
        self,
        i,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=300,
        batch_size=128,
        n=None,
        shuffle=False,
    ):
        # Transform the input
        X_train, X_test = self.transform(X_train, X_test, self.enable)

        # Train the model
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=0.005))
        self.model.fit(
            X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
        )

        # Evaluate the model
        X_train_pred = self.model.predict(X_train)

        # Plot the results
        filename = self.model_name + "-" + str(i)
        plot_autoencoder(X_train, X_train_pred, n, filename)

    def bottleneck(self, X_train, X_test):
        standalone_encoder = Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[self.bottleneck_layer].output,
        )

        X_train_features = standalone_encoder.predict(X_train)
        X_train_features = X_train_features.squeeze()

        return X_train_features

    def reset_weights(self):
        for i, layer in enumerate(self.model.layers):
            if hasattr(self.model.layers[i], "kernel_initializer") and hasattr(
                self.model.layers[i], "bias_initializer"
            ):
                weight_initializer = GlorotUniform()
                bias_initializer = Zeros()
                old_weights, old_biases = self.model.layers[i].get_weights()

                self.model.layers[i].set_weights(
                    [
                        weight_initializer(shape=old_weights.shape),
                        bias_initializer(shape=old_biases.shape),
                    ]
                )

    @staticmethod
    def transform(X_train, X_test, enable=False):
        if enable:
            X_train = np.reshape(X_train, X_train.shape + (1,))
            X_test = np.reshape(X_test, X_test.shape + (1,))

        return X_train, X_test


class LSTMAutoEncoder(AutoEncoder):
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=3, drop_prob=0.3):
        self.model_name = "lstm_autoencoder"
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
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=3, drop_prob=0.3):
        self.model_name = "basic_autoencoder"
        self.bottleneck_layer = 0
        self.enable = False

        model = Sequential()
        model.add(Input(shape=(timesteps,)))
        model.add(Dense(encoding_dim, activation=LeakyReLU()))
        model.add(Dense(timesteps, activation="sigmoid"))

        self.model = model
        self.model.summary()


class DeepAutoEncoder(AutoEncoder):
    def __init__(self, timesteps=30, input_dim=1, encoding_dim=3, drop_prob=0.3):
        self.model_name = "deep_autoencoder"
        self.bottleneck_layer = 2
        self.enable = False

        model = Sequential()
        model.add(Input(shape=(timesteps,)))
        model.add(Dense(32, activation=LeakyReLU()))
        model.add(BatchNormalization())
        model.add(Dense(encoding_dim, activation=LeakyReLU()))
        model.add(Dense(32, activation=LeakyReLU()))
        model.add(BatchNormalization())
        model.add(Dense(timesteps, activation="sigmoid"))

        self.model = model
        self.model.summary()
