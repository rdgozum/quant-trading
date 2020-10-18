import pandas as pd
import pandas_datareader as dr

from datetime import datetime

from quant_trading import settings


class StockDataset:
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(self):
        self.indicators = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]

    @property
    def symbols(self):
        df = pd.read_html(self.URL)[0]

        return df.loc[:, "Symbol"]

    @property
    def sectors(self):
        df = pd.read_html(self.URL)[0]

        return df.loc[:, "GICS Sector"]

    def get_sector(self, symbol):
        df = pd.read_html(self.URL)[0]

        filter = df["Symbol"] == symbol
        sector = df.loc[filter, "GICS Sector"].item()

        return sector

    def extract_range(
        self, start_date, end_date, indicators=None, symbols=None, drop_threshold=0.90
    ):
        if indicators is None:
            indicators = self.indicators
        if symbols is None:
            symbols = self.symbols

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        list = []
        for symbol in symbols:
            symbol_df = dr.DataReader(symbol, "yahoo", start_date, end_date)[indicators]
            symbol_df.columns = [symbol + " " + i for i in indicators]
            list.append(symbol_df)

            df = pd.concat(list, axis=1)

        df.dropna(axis="columns", thresh=df.shape[0] * drop_threshold, inplace=True)
        df.sort_index(axis="columns", inplace=True)

        return df

    @staticmethod
    def output_reader(filename):
        path = settings.results(filename)
        df = pd.read_csv(path)

        return df

    @staticmethod
    def output_writer(df, filename):
        path = settings.results(filename)
        df.to_csv(path)
