import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import os

from data_preprocessing import data_preprocessing
import prepare_data_for_supervised as pdfs

#-----------------------
# original data for vwap
if not os.path.exists('./newdata'):
    data_preprocessing()

df0 = pd.read_csv('./newdata/goods/0_goods.csv', index_col=0, parse_dates = True)

# normalization
scaled_df, scaler = pdfs.scaled_dataframe(df0)
# more parameters
n_days, n_features = 1, 3
        # vwap, volume, return
# dataset for supervised learning
dataset_sup = pdfs.prepare_for_supervised(scaled_df, n_days, 1)
# the last feature is the label ! history label also is a feature.

train_X, train_y, test_X, test_y = pdfs.split_train_test(dataset_sup.values, n_days, n_features, split_ratio=0.8)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# LSTM network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
# hidden layer neurons
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=30, batch_size=150, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
# inverse_scale
inv_yhat = pdfs.inverse_scale(scaler, test_X, yhat, n_features)
# inverse scaling for test_y
inv_y = pdfs.inverse_scale(scaler, test_X, test_y, n_features)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#print(yhat, test_y)
plt.figure(figsize=(15,8))
plt.plot(list(range(len(inv_y))), inv_y,'.-',label='y-real')
plt.plot(list(range(len(inv_yhat))), inv_yhat,'-', label='y-predict')
plt.legend()
plt.show()
