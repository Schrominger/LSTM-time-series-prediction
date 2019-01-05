import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_for_supervised(dataF, n_in=1, n_out=1, dropnan=True):
    """
    dataF: original DataFrame
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    returns: Pandas DataFrame for supervised learning.
    """

    n_vars = dataF.shape[1]

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dataF.shift(i))
        names += [(str(col)+'(t-%d)' % i) for col in dataF.columns]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dataF.shift(-i))
        if i == 0:
            names += [(str(col)+'(t)') for col in dataF.columns]
        else:
            names += [(str(col)+'(t+%d)' % i) for col in dataF.columns]
    # put it all together
    df = pd.concat(cols, axis=1)
    df.columns = names
    # drop rows with NaN values
    if dropnan:
        df.dropna(inplace=True)
    return df

def scaled_dataframe(df):
    # normalize features
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index = df.index, columns=df.columns)
    # need to return the scaler for recover the data!
    return scaled_df, scaler

def inverse_scale(scaler, X, y, n_features):
    # invert scaling for prediction
    # 这里随便给n_features-1列放在前面做 inverse_transform，其实只要最后一列对就行
    inv_y = np.concatenate((X[:,-n_features:-1], y.reshape(-1,1)), axis=1)
    inv_y = scaler.inverse_transform(inv_y)[:, -1]
    return inv_y


def split_train_test(sup_df_values, n_days, n_features, split_ratio=0.67):
    '''
    sup_df_values: DataFrame, only last column is the label y(t)
    return: train_X, train_y   test_X, test_y          [np.array]
    '''
    M, N = sup_df_values.shape
    train_size = int(M*split_ratio)

    train = sup_df_values[:train_size, :]
    test = sup_df_values[train_size:, :]

    n_obs = n_days*n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]

    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    values = pd.DataFrame(np.random.randn(10,2), columns=list('ab'))
    data = prepare_for_supervised(values, n_in=2)
    print(data, '\n')
    print('Initial df：\n', values)
