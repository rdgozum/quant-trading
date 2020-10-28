# Quant Trading

Python trading and backtesting platform.

## About
- Utilizes *pandas_datareader* interface to access stock information about the S&P 500 companies from yahoo finance.
- Trains an AutoEncoder to perform dimensionality reduction (i.e. learn the compressed representations of time series samples).
- Uses KNN and DBSCAN algorithms to discover clusters from the samples to help us identify potentially good matches for pairs trading strategy later.
- **TODOs**: Trading and Backtesting

## Prerequisites
Run the commands below to setup your local environment.

```bash
$ git clone git@github.com:rdgozum/quant-trading.git
$ cd quant-trading
$ pip install -r requirements.txt
```

## Project Usage
```bash
usage: python main.py [-h] [-t1 START_DATE] [-t2 END_DATE] [--do-extract-save]
                    [--do-extract-from-file] [--do-training] [--do-similarity]
                    [--find-epsilon] [--model MODEL] [--epochs EPOCHS]
                    [--batch-size BATCH_SIZE] [--encoding-dim ENCODING_DIM]
                    [--epsilon EPSILON] [--n N]
```

#### Extract/Save Web Data to File
- Pull stock information between `START_DATE` and `END_DATE` (format: yyyy-mm-dd) from the web using an API from *pandas_datareader*. The data will be saved as pandas dataframe to results folder.

Example:
```bash
python main.py [-t1 START_DATE] [-t2 END_DATE] --do-extract-save
```

#### Read Data from File
- Read a saved data from results folder as pandas dataframe.

Example:
```bash
python main.py [-t1 START_DATE] [-t2 END_DATE] --do-extract-from-file
```

#### Train an AutoEncoder
- Train an AutoEncoder to perform dimensionality reduction on time series dataset. Options for the model include `lstm_autoencoder`, `basic_autoencoder`, and `deep_autoencoder`.

Example:
```bash
python main.py  [-t1 START_DATE] [-t2 END_DATE] [--do-extract-save] [--do-extract-from-file]
               --do-training basic_autoencoder
               --batch-size 128 --epochs 300 --encoding-dim 3
```


#### Feature Similarities
- After training, we can calculate similarities (or distances) between the feature vectors in low dimension that will eventually allow us to discover clusters. We can use `--find-epsilon` first to help us identify an optimal epsilon parameter for DBSCAN using the *elbow method*.

Example:
```bash
python main.py [-t1 START_DATE] [-t2 END_DATE] --do-similarity --find-epsilon [--encoding-dim ENCODING_DIM]

OR

python main.py [-t1 START_DATE] [-t2 END_DATE] --do-similarity --epsilon 5 [--encoding-dim ENCODING_DIM]
```
