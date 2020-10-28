import pandas as pd
from quant_trading.datasets import stock_dataset

from quant_trading import settings


def write_similarity(encoding_dim, symbols, cluster_labels, distances, indices):
    stock_data = stock_dataset.StockDataset()

    print("cluster_labels:\n", cluster_labels)
    print("indices:\n", indices)

    similarity_output = []
    for i in range(10):  # len(indices)
        print(f"Processing index {i}")
        for j in indices[i]:
            symbol = symbols[j].replace("-", ".")
            sector, subindustry = stock_data.get_industry_from_symbol(symbol)

            similarity_output.append(
                {
                    "index": i,
                    "neighbor": j,
                    "symbol": symbol,
                    "sector": sector,
                    "subindustry": subindustry,
                    "cluster": cluster_labels[j],
                }
            )

    df = pd.DataFrame(similarity_output)
    df_path = settings.results(f"dim{encoding_dim}-similarity_output.csv")

    df.to_csv(df_path)
