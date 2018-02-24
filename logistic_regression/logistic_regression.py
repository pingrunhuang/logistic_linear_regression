import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression

'''
logistic regression for classification
Like telling if the email is spam or not spam
This implementation is based on multi features.
'''


def get_data(src='dataset/breast-cancer-wisconsin.data', out='breast_cancer.pickle'):
    df = pd.read_csv(src)
    with open(out, 'wb') as file:
         pickle.dump(df, file)


def sigmoid(x, theta):
    '''
    The idea of a sigmoid function is go penalize the error classification with a great cost
    '''
    return (1.0 / (1.0 + np.exp(-(x.dot(theta))))).astype('float')

def cost_function(x, y, theta):
    '''
    Choosing a cost function should always consider if it is convex or not
    In logistic regression, it is different from linear regression
    '''
    hx = sigmoid(x, theta)
    const = -1/y.count()
    return const * np.sum(np.log(hx).dot(y) + np.log(1-hx).dot(1-y))

def validation(testX, testY, theta):

    skscore = classifier.score(testX, testY)
    myscore = 0
    hx = sigmoid(testX, theta)
    index = 0
    for result in hx:
        if result > 0.5 and testY.iloc[index] == 1:
            myscore += 1
        if result <=0.5 and testY.iloc[index] == 0:
            myscore += 1
        index += 1
    myscore = float(myscore / testY.count())
    print('my score:' , myscore)
    print('sk score:', skscore)

def gradient_descent(trainX, trainY, testX, testY, num_iter=1000):
    cost = 10
    count = 0
    theta = np.random.rand(trainX.iloc[0].count())
    for _ in range(num_iter):
        # partial derivitive of cost function
        theta = theta - (sigmoid(trainX, theta) - trainY).transpose().dot(trainX) / trainY.count()
        cost = cost_function(trainX, trainY, theta=theta)
        print("Iteration: ", count)
        print('theta:',theta)
        print('cost:', cost)
        count += 1

    validation(testX, testY, theta)


def regularized_cost_function():

    pass

def regularized_cost_function():

    pass
def lr_with_conjugate_gradient():
    from scipy.optimize import fmin_cg
    pass

src = 'dataset/breast-cancer-wisconsin.data'
pickle_data = 'breast_cancer.pickle'
if os.path.isfile(src):
    get_data()
df = pd.read_pickle(pickle_data)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

test_ratio = 0.3
train_dataset, test_dataset = train_test_split(df, test_size=test_ratio)
trainX = train_dataset.iloc[:,1:9].astype(float)
trainX['x0'] = 1
trainY = train_dataset.iloc[:,10]
trainY = trainY/2 - 1
testX = test_dataset.iloc[:,1:9].astype(float)
testX['x0'] = 1
testY = test_dataset.iloc[:,10]


classifier = LinearRegression()
classifier.fit(trainX, trainY)

gradient_descent(trainX, trainY, testX, testY)
