import argparse


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
        "--find-optimal-epsilon",
        dest="find_optimal_epsilon",
        action="store_true",
        help="determine the optimal epsilon from k-distance elbow plot",
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
        "--epsilon",
        dest="epsilon",
        type=float,
        help="radius of neighborhood around the points during clustering",
    )
    parser.add_argument(
        "--n", dest="n", type=int, help="number of windows to display",
    )

    return parser.parse_args()
