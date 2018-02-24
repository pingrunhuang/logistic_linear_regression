import pickle
import pandas as pd
import numpy as np
import sympy
from sklearn.model_selection import train_test_split

'''
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's

 Notice this script is implementing multiple variables linear regression
 So we will get n theta eventually
'''

df = pd.read_pickle('dataset/housing_dataset.pickle')

test_ratio = 0.3
train_dataset, test_dataset = train_test_split(df, test_size=test_ratio)
trainX = train_dataset.iloc[:,:13]
trainY = train_dataset.iloc[:,13]
testX = test_dataset.iloc[:,:13]
testY = test_dataset.iloc[:,13]


'''
remember regularization is to solve overfitting problem
'''
def regularization_cost_function(x, y, theta, theta0, lamda):
    '''
    J(theta) = [sum(square(theta * x - y)) + lamda * sum(square(theta))] / (2 * m)
    params:
    x is in the form of (1, x1, x2, ... , xn)
    y is the actual result
    theta is in the form (theta0, theta1, ... , thetan)
    '''
    hx = x.dot(theta.transpose()) + theta0
    return (np.sum(np.square(hx - y)) + lamda * np.sum(np.square(theta))) / (2 * y.count())

def gradient_descent_with_regularization(trainX, trainY, lamda=0.01):
    '''
    The goal of regularization is to minimize the theta affectness
    '''
    MSE=10
    trainX = (trainX-trainX.mean()).divide(trainX.max()-trainX.min())
    trainX['x0'] = 1
    theta = np.random.rand(trainX.iloc[0].count())
    alpha = 0.01
    count = 0
    # the lamda array is [0.01, 0.01, ..., 0]
    # we are doing so because the parameter descent is different between theta0 and others
    lamda_array = np.empty(trainX.iloc[0].count()-1)
    lamda_array.fill(lamda)
    np.append(lamda, 0)
    while MSE > 0.05:
        # this equation is based on the partial derivitive of J(theta) over theta
        theta = theta - alpha * ((trainX.dot(theta) - trainY).dot(trainX) + lamda_array.dot(theta)) / trainY.count()
        MSE = regularization_cost_function(trainX, trainY, theta, theta0, lamda)
        print('Mean squared error:', MSE)
        print('Iteration:', count)
        count = count + 1




def gradient_descent(trainX, trainY):
    # it will converge faster if we subscribe the average value then divide the range of each column
    # this is the key step for convergence
    trainX = (trainX-trainX.mean()).divide(trainX.max()-trainX.min())
    # x = [x1, x2, x3 ... 1]
    trainX['x0'] = 1
    # question: if we need to normalize the y?
    # trainY = (trainY-trainY.mean()).divide(trainY.max()-trainY.min())
    MSE = 10
    theta = np.random.rand(trainX.iloc[0].count())
    count = 0
    alpha = 0.01

    while MSE > 4.6:
        # h(x) = theta0 + theta1 * x
        y_estimate = trainX.dot(theta.transpose())
        # update simultaneously
        # I use * here instead of .dot because each column is multiplying the same value which is the corresponding h(x) - y
        # I am pretty sure this function goes the same as the definition from coursera class taught by Andrew, but somehow it get negative
        theta = theta - alpha * ((y_estimate - trainY) * trainX.transpose()).sum(axis=1)/trainY.count()
        # update MSE: 1/2 * 1/m * sum(square(h(x(i)) - y(i)))
        # since the theta1 is negative, the MSE is getting larger and larger why?
        # MSE = np.sum(np.square(theta0 + trainX.dot(theta1)-trainY)) / ( 2 * trainY.count())
        MSE = np.sqrt(np.sum(np.square(trainX.dot(theta)-trainY)) / trainY.count())
        print('Current iteration: ' + str(count))
        # print('h(x):' + str(theta0 + trainX.dot(theta1)))
        # print('y:' + str(trainY))
        print('Mean squared error:', MSE)
        count = count + 1

    # validation
    # I am not sure if I am doing it right, but the tesing error is about 25250 which is pretty large
    MSE = np.sqrt(np.sum(np.square(theta0 + testX.dot(theta1)-testY)) / testY.count())
    print("The testing MSE is: " + str(MSE))

def normal_equation():
    trainX = train_dataset.iloc[:,:13]
    trainX['x0'] = 1
    trainY = train_dataset.iloc[:,13]
    testX = test_dataset.iloc[:,:13]
    testX['x0'] = 1
    testY = test_dataset.iloc[:,13]
    theta = np.linalg.pinv(trainX.transpose().dot(trainX)).dot(trainX.transpose()).dot(trainY)
    MSE = np.sqrt(np.sum(np.square(testX.dot(theta) - testY)) / testY.count())
    print(MSE)

# normal_equation()
gradient_descent(trainX, trainY)
# sklearn_linear_regression(trainX, trainY, testX, testY)
# gradient_descent_with_regularization(trainX, trainY)
