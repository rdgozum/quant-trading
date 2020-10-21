"""Main module."""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(".."))

from quant_trading.datasets.stock_dataset import StockDataset
from quant_trading.datasets import preprocessor
from quant_trading.models.autoencoders import (
    BasicAutoEncoder,
    LSTMAutoEncoder,
    DeepAutoEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default="2015-01-01",
        help="extracts data from the specified start-date (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        default="2015-12-01",
        help="extracts data until the specified end-date (default: 2015-12-01)",
    )
    parser.add_argument(
        "--do-extract-save",
        dest="do_extract_save",
        action="store_true",
        help="extracts and saves the pulled range to csv file",
    )
    parser.add_argument(
        "--do-extract-from-file",
        dest="do_extract_from_file",
        action="store_true",
        help="extracts the pulled range data from csv file",
    )
    parser.add_argument(
        "--do-training",
        dest="do_training",
        action="store_true",
        help="perform model training",
    )
    parser.add_argument(
        "--do-clustering",
        dest="do_clustering",
        action="store_true",
        help="perform clustering",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="basic_autoencoder",
        help="the model architecture to be trained (default: basic_autoencoder)",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=100,
        type=int,
        help="number of complete pass through the training data (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=128,
        type=int,
        help="number of training examples utilized in one iteration (default: 128)",
    )
    parser.add_argument(
        "--n", dest="n", type=int, help="number of windows to display",
    )

    return parser.parse_args()


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

        print("Save to csv file...")
        stock_data.output_writer(df, f"data-{start_date}_{end_date}.csv")

    if args.do_extract_from_file:
        print("Read from csv file...")
        df = stock_data.output_reader(f"data-{start_date}_{end_date}.csv")
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

        model.train(
            X_train[0],
            y_train[0],
            X_test[0],
            y_test[0],
            epochs=args.epochs,
            batch_size=args.batch_size,
            n=args.n,
        )

        X_train_features = model.pull_bottleneck(X_train[0], X_test[0])
        print("X_train_features")
        print(X_train_features.shape)
        print(X_train_features)

    # Clustering
    if args.do_clustering:
        pass


if __name__ == "__main__":
    args = parse_args()

    run(args)
