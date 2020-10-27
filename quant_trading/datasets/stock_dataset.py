import pickle
import pandas as pd
import pandas_datareader as dr

from datetime import datetime

from quant_trading import settings


class StockDataset:
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(self):
        self.indicators = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
        self._symbols = self.read_df(field="Symbol")

    @property
    def symbols(self):
        return self._symbols

    @symbols.setter
    def symbols(self, value):
        self._symbols = value

    def get_sector_from_symbol(self, symbol):
        df = pd.read_html(self.URL)[0]
        print(df)

        filter = df["Symbol"] == symbol
        sector = df.loc[filter, "GICS Sector"].item()

        return sector

    def read_df(self, field):
        df = pd.read_html(self.URL)[0]

        return df.loc[:, field].tolist()

    def extract_range(
        self, start_date, end_date, indicators=None, symbols=None, drop_threshold=0.90
    ):
        if indicators is None:
            indicators = self.indicators
        if symbols is None:
            symbols = self._symbols

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        list = []
        for symbol in symbols:
            try:
                symbol = symbol.replace(".", "-")
                symbol_df = dr.DataReader(symbol, "yahoo", start_date, end_date)[
                    indicators
                ]
                symbol_df.columns = [symbol + " " + i for i in indicators]
                list.append(symbol_df)

                df = pd.concat(list, axis=1)
            except:
                pass

        df.dropna(axis="columns", thresh=df.shape[0] * drop_threshold, inplace=True)
        df.sort_index(axis="columns", inplace=True)

        return df

    def output_reader(self, date_range):
        # reading dataframe
        df_path = settings.results(f"df-{date_range}.csv")
        df = pd.read_csv(df_path)

        # reading symbols
        symbols_path = settings.results(f"symbols-{date_range}.pkl")
        with open(symbols_path, "rb") as pickle_file:
            symbols = pickle.load(pickle_file)

        return df, symbols

    def output_writer(self, df, date_range):
        # saving dataframe
        df_path = settings.results(f"df-{date_range}.csv")
        df.to_csv(df_path)

        # saving symbols
        symbols_path = settings.results(f"symbols-{date_range}.pkl")
        with open(symbols_path, "wb") as pickle_file:
            pickle.dump(self._symbols, pickle_file)
