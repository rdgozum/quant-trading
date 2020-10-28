import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t1",
        "--start-date",
        dest="start_date",
        default="2016-01-01",
        help="start date (default: 2016-01-01)",
    )
    parser.add_argument(
        "-t2",
        "--end-date",
        dest="end_date",
        default="2020-01-01",
        help="end date (default: 2020-01-01)",
    )
    parser.add_argument(
        "--do-extract-save",
        dest="do_extract_save",
        action="store_true",
        help="extract/save web data to file",
    )
    parser.add_argument(
        "--do-extract-from-file",
        dest="do_extract_from_file",
        action="store_true",
        help="read data from file",
    )
    parser.add_argument(
        "--do-training",
        dest="do_training",
        action="store_true",
        help="train an autoencoder to extract features",
    )
    parser.add_argument(
        "--do-similarity",
        dest="do_similarity",
        action="store_true",
        help="calculate feature similarities",
    )
    parser.add_argument(
        "--find-epsilon",
        dest="find_epsilon",
        action="store_true",
        help="find optimal epsilon value for dbscan",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="basic_autoencoder",
        help="autoencoder type (default: basic_autoencoder)",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=300,
        type=int,
        help="number of epochs (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=128,
        type=int,
        help="training batch size (default: 128)",
    )
    parser.add_argument(
        "--epsilon", dest="epsilon", type=float, help="epsilon value for dbscan",
    )
    parser.add_argument(
        "--n", dest="n", type=int, help="number of windows to display",
    )

    return parser.parse_args()
