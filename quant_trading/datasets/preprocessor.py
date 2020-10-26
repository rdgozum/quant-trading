import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Preprocessors
def normalize(columns, df_train, df_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train = scaler.fit_transform(df_train[columns])
    df_test = scaler.fit_transform(df_test[columns])

    return df_train, df_test


def reshape(df_train, df_test):
    df_train = np.asarray(df_train).astype(np.float32)
    df_test = np.asarray(df_test).astype(np.float32)

    df_train = df_train.T
    df_test = df_test.T

    return df_train, df_test


def temporalize(X, timesteps):
    Xs, ys = [], []
    for i in range(len(X)):
        vector_x, vector_y = [], []
        for j in range(len(X[i]) - timesteps):
            v = X[i][j : (j + timesteps)]
            vector_x.append(v)
            vector_y.append(X[i][j + timesteps])
        Xs.append(vector_x)
        ys.append(vector_y)
    return np.array(Xs), np.array(ys)


def run(df):
    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

    # Normalize
    columns = df.columns.tolist()
    df_train, df_test = normalize(columns, df_train, df_test)

    # Reshape
    df_train, df_test = reshape(df_train, df_test)

    # Temporalize
    X_train, y_train = temporalize(df_train, timesteps=df_train.shape[1] - 1)
    X_test, y_test = temporalize(df_test, timesteps=df_train.shape[1] - 1)

    return X_train, y_train, X_test, y_test
