"""Main module."""
import os
import sys
import numpy as np
import pickle


sys.path.insert(0, os.path.abspath(".."))

from quant_trading.datasets.stock_dataset import StockDataset
from quant_trading.datasets import preprocessor
from quant_trading.models.autoencoders import (
    BasicAutoEncoder,
    LSTMAutoEncoder,
    DeepAutoEncoder,
)
from quant_trading.visualization.clustering import (
    DBSCANClustering,
    optimal_epsilon,
    optimal_min_samples,
)

from quant_trading import config, settings


def run(args):
    stock_data = StockDataset()
    start_date = args.start_date
    end_date = args.end_date

    # Dataset
    if args.do_extract_save:
        symbols = None
        indicators = ["Close"]

        print("Extract from {} to {}...".format(start_date, end_date))
        df = stock_data.extract_range(start_date, end_date, indicators, symbols)
        print(df.head())

        symbols = []
        for i in df.columns.tolist():
            x = i.split(" ")
            if x[0] not in symbols:
                symbols.append(x[0])
        stock_data.symbols = symbols

        print("Save to csv file...")
        stock_data.output_writer(df, f"{start_date}_{end_date}")

    if args.do_extract_from_file:
        print("Read from csv file...")
        df, symbols = stock_data.output_reader(f"{start_date}_{end_date}")
        df = df.set_index("Date")
        print(df.head())

        X_train, y_train, X_test, y_test = preprocessor.run(df)
        print(f"X_train: {type(X_train)}, y_train: {type(y_train)}")
        print(X_train.shape, y_train.shape)
        print(f"X_test: {type(X_test)}, y_test: {type(y_test)}")
        print(X_test.shape, y_test.shape)

    # Feature Extraction
    if args.do_training:
        print("Start training...")
        if args.model == "basic_autoencoder":
            model = BasicAutoEncoder(timesteps=X_train.shape[2])
        if args.model == "deep_autoencoder":
            model = DeepAutoEncoder(timesteps=X_train.shape[2])
        if args.model == "lstm_autoencoder":
            model = LSTMAutoEncoder(timesteps=X_train.shape[2])

        features = []
        for i in range(X_train.shape[0]):
            model.train(
                i,
                X_train[i],
                y_train[i],
                X_test[i],
                y_test[i],
                epochs=args.epochs,
                batch_size=args.batch_size,
                n=args.n,
            )
            X_train_features = model.bottleneck(X_train[i], X_test[i])
            model.reset_weights()

            features.append(X_train_features)

        filename = args.model
        with open(settings.results(f"{filename}-features.pkl"), "wb") as pickle_file:
            pickle.dump(features, pickle_file)

    # Clustering
    if args.do_clustering:
        filename = args.model
        with open(settings.results(f"{filename}-features.pkl"), "rb") as pickle_file:
            features = pickle.load(pickle_file)

        with open(
            settings.results(f"symbols-{start_date}_{end_date}.pkl"), "rb"
        ) as pickle_file:
            symbols = pickle.load(pickle_file)

        features = np.asarray(features, dtype=np.float32)
        min_samples = optimal_min_samples(features)
        if args.find_optimal_epsilon:
            optimal_epsilon(features, min_samples)
        else:
            dbscan = DBSCANClustering(args.epsilon, min_samples)
            dbscan.run(features, symbols)


if __name__ == "__main__":
    args = config.parse_args()

    run(args)
