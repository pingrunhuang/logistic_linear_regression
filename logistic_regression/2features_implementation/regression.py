from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,7,6], dtype=np.float64)

# generate fake data for testing
def create_data(hm, variance, step = 2, correlation='pos'):
    # y axis start from 1
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def get_best_fit_slop_and_intercept(xs, ys):
    # m = (mean(xs) * mean(ys) - mean(xs * ys)) / mean(xs)**2 - mean(xs**2)
    m = (mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs) * mean(xs) - mean(xs * xs))
    b = mean(ys) - m * mean(xs)
    return m, b

def draw(m, b, x_set):
    line = [(m * x + b) for x in x_set]

def predicit(x, regression_function, x_set, y_set):
    m, b = regression_function(x_set, y_set)
    return m * x + b

'''
squared_error = (y_predict - y_origin)^2
coefficient_of_determination = 1 - (squared_error(y_predict) / squared_error(y_mean))
'''

# input types are 2 lists
def squared_error(y_origin, y_predict):
    return sum((y_origin - y_predict)**2)

# the closer the result is to 1, the better regression
def coefficient_of_determination(y_predict, y_origin):
    y_mean = [mean(y_origin) for _ in y_origin]
    y_predict_squared_error = squared_error(y_predict, y_origin)
    y_mean_squared_error = squared_error(y_mean, y_origin)
    return 1 - (y_predict_squared_error / y_mean_squared_error)


'''
test assumption
'''

xs, ys = create_data(40, 10, 2, correlation='neg')

m, b = get_best_fit_slop_and_intercept(xs, ys)
y_predict = [m * x + b for x in xs]


style.use('fivethirtyeight')
plt.plot(y_predict)
error = coefficient_of_determination(y_predict, ys)
plt.xlabel(error)
plt.scatter(xs, ys)
plt.show()


