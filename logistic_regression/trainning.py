from sklearn.linear_model import LinearRegression
import pickle
from .DBUtil import *
'''
tranning the model and saving it as a pickle file could be done on the cloud
so that we do not need to train the model every time
'''

# train the model
# n_jobs: The number of jobs to use for the computation. If -1 all CPUs are used
classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train, Y_train)
# save the classifier

with open("classifier.pickle", 'wb') as file:
    pickle.dump(classifier, file)
