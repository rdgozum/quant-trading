"""Main module."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from quant_trading.datasets import stock_dataset, preprocessor, dataset_utils
from quant_trading.models import autoencoders, model_utils
from quant_trading.visualizations.clustering import (
    DBSCANClustering,
    optimal_epsilon,
    optimal_min_samples,
)
from quant_trading import config, settings


def run(args):
    start_date = args.start_date
    end_date = args.end_date

    # Dataset
    if args.do_extract_save:
        stock_data = stock_dataset.StockDataset()
        symbols = None
        indicators = ["Close"]

        df = stock_data.extract_range(start_date, end_date, indicators, symbols)
        dataset_utils.write_df(df, start_date, end_date)
        dataset_utils.write_symbols(stock_data.symbols, start_date, end_date)

    if args.do_extract_from_file:
        df = dataset_utils.read_df(start_date, end_date)
        symbols = dataset_utils.read_symbols(start_date, end_date)

    # Feature Extraction
    if args.do_training:
        X_train, y_train, X_test, y_test = preprocessor.run(df)

        if args.model == "basic_autoencoder":
            model = autoencoders.BasicAutoEncoder(timesteps=X_train.shape[2])
        if args.model == "deep_autoencoder":
            model = autoencoders.DeepAutoEncoder(timesteps=X_train.shape[2])
        if args.model == "lstm_autoencoder":
            model = autoencoders.LSTMAutoEncoder(timesteps=X_train.shape[2])

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

        model_utils.write_features(features, args.model, start_date, end_date)

    # Clustering
    if args.do_clustering:
        features = model_utils.read_features(args.model, start_date, end_date)
        symbols = dataset_utils.read_symbols(start_date, end_date)

        min_samples = optimal_min_samples(features)
        if args.find_optimal_epsilon:
            optimal_epsilon(features, min_samples)
        else:
            dbscan = DBSCANClustering(args.epsilon, min_samples)
            dbscan.run(features, symbols)


if __name__ == "__main__":
    args = config.parse_args()

    run(args)
