from matplotlib import style
import matplotlib.pyplot as plt
import pickle, datetime
import numpy as np
import DBUtil
'''
on the previous section, we are using linear regression as classifier
next, we will use it for prediction
'''

def sklearn_linear_regression(x_train, y_train, x_test, y_test):
    '''
    using
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    classifier = LinearRegression(n_jobs=-1)
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)
    print("Accuracy: ", accuracy)
    predict = classifier.predict(testX)
    print("MSE: ", mean_squared_error(testY, predict))

with open("classifier.pickle", 'rb') as file:
    classifier = pickle.load(file)
X_lately = DBUtil.X[-DBUtil.forecast_out:]
X = DBUtil.X[:-DBUtil.forecast_out]
accuracy = classifier.score(DBUtil.X_test, DBUtil.Y_test)
print('Accuracy score:', accuracy)
forecast_set = classifier.predict(X_lately)
print(forecast_set)
DBUtil.df['Forecast'] = np.nan

# initialization
# the last row's name
last_date = DBUtil.df.iloc[-1].name
last_unix_time = last_date.timestamp()
# one day contains 86400 sec
one_day = 86400
next_unix_time = last_unix_time + one_day


'''
the goal is to get a dataframe looks like this
    X       Nan
    Nan     forecast_set
'''
for entry in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix_time)
    next_unix_time += one_day
    # horizontally append the cell value
    DBUtil.df.loc[next_date] = [np.nan for _ in range(len(DBUtil.df.columns) - 1)] + [entry]


# plot part
style.use('ggplot')
DBUtil.df['Adj. Close'].plot()
DBUtil.df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
# plt.legend(loc=4)
plt.show()
