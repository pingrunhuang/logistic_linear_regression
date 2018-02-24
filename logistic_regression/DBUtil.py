import quandl as qd
import numpy as np
import math
from sklearn import preprocessing, model_selection

# get data frame containing the data of google stock price from the quandl
df = qd.get('WIKI/GOOGL')
# filter
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
# normalize
df['HL_Pct'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['Pct_change'] = (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# continue filter
df = df[['Adj. Close', 'HL_Pct', 'Pct_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
# replace the null data with -99999, this can avoid some error
df.fillna(-99999, inplace=True)
# number of rows we are going to shift up. it is also the day we are going to predict
forecast_out = int(math.ceil(0.01 * len(df)))
# shifting the forecast_col up with forecast_out rows
df['label'] = df[forecast_col].shift(-forecast_out)
# remove all the rows with null value that are caused by the shifting function
df.dropna(inplace=True)

# in order to do normalizing data, have to convert a column to an array using numpy
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
Y = np.array(df['label'])

# this is how to split the data into train and test data, called the cross validation
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)