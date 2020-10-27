import pandas as pd
import pandas_datareader as dr
from datetime import datetime


class StockDataset:
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(self):
        self.indicators = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
        self.symbols = self._read_df(field="Symbol")

    def _read_df(self, field):
        df = pd.read_html(self.URL)[0]

        return df.loc[:, field].tolist()

    def get_sector_from_symbol(self, symbol):
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

        self._update_symbols(df)

        return df

    def _update_symbols(self, df):
        symbols = []
        for i in df.columns.tolist():
            x = i.split(" ")
            if x[0] not in symbols:
                symbols.append(x[0])
        self.symbols = symbols
