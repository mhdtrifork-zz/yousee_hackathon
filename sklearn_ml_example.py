# coding=utf-8
#based on the follwing tutorial: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')


# print data.head()
# print data.shape
# print data.describe()

y = data.quality
X = data.drop('quality', axis=1)


#divide into test and train sets
# (test_size is the % of the total sample to use for validation,
# random_state makes sure that it is split in the same way, every time it is run with the same number,
# stratify means to make sure that you get a good sample of all the different groups
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# standardize data
# save means and standard deviations, so the standarditation process is the same for the test and train sets
#scaler = preprocessing.StandardScaler().fit(X_train)
#this is done automatically in the crossvalidation pipeline here:
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

#tuning parameters (docs: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html):
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

#perform crossvalidation on the set by cutting it into cv chunks and
#validate 1 chunk against the rest, cv times
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# Fit and tune model
clf.fit(X_train, y_train)


print clf.best_params_
print clf.refit

y_pred = clf.predict(X_test)

print y_pred
print y_test
#R^2 (coefficient of determination) regression score function.
#Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
# A constant model that always predicts the expected value of y, #
# disregarding the input features, would get a R^2 score of 0.0.
print r2_score(y_test, y_pred)

#The MSE is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.
print mean_squared_error(y_test, y_pred)