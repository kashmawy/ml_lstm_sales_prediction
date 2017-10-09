from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
 
# date-time parsing function for loading the dataset
def date_parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def scale(train, test):
	scaler = MinMaxScaler(feature_range=(-1,1))
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	import pytest; pytest.set_trace()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		if (i % 100) == 0:
			print("Epochs is at %d" % i)
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0, 0]

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=date_parser)
raw_values = series.values
diff_values = difference(raw_values, 1)
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
train, test = supervised_values[0:-12], supervised_values[-12:]

import pytest; pytest.set_trace()

scaler, train_scaled, test_scaled = scale(train, test)
import pytest; pytest.set_trace()
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

predicts = list()
for i in range(len(test_scaled)):
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	yhat = invert_scale(scaler, X, yhat)
	yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
	predicts.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

rmse = sqrt(mean_squared_error(raw_values[-12:], predicts))
print('Test RMSE: %.3f' % rmse)

print("Going to plot")
pyplot.plot(raw_values[-12:])
pyplot.plot(predicts)
pyplot.show()
