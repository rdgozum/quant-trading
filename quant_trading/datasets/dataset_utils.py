import pandas as pd
import pickle

from quant_trading import settings


def read_df(start_date, end_date):
    date_range = f"{start_date}_{end_date}"
    df_path = settings.results(f"df-{date_range}.csv")

    df = pd.read_csv(df_path)
    df = df.set_index("Date")

    return df


def read_symbols(start_date, end_date):
    date_range = f"{start_date}_{end_date}"
    symbols_path = settings.results(f"symbols-{date_range}.pkl")

    with open(symbols_path, "rb") as pickle_file:
        symbols = pickle.load(pickle_file)

    return symbols


def write_df(df, start_date, end_date):
    date_range = f"{start_date}_{end_date}"
    df_path = settings.results(f"df-{date_range}.csv")

    df.to_csv(df_path)


def write_symbols(symbols, start_date, end_date):
    date_range = f"{start_date}_{end_date}"
    symbols_path = settings.results(f"symbols-{date_range}.pkl")

    with open(symbols_path, "wb") as pickle_file:
        pickle.dump(symbols, pickle_file)
