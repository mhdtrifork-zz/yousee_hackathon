# coding=utf-8
#based on the follwing tutorial: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

import numpy as np
import pandas as pd
from sklearn.externals import joblib

dataset_url = 'Test_dataset.csv'
data = pd.read_csv(dataset_url, sep=';')
data['TV_technology'], unique = pd.factorize(data['TV_technology'])
data['CZ_group5_name'], unique = pd.factorize(data['CZ_group5_name'])
data['TV_product_grouped'], unique = pd.factorize(data['TV_product_grouped'])
data['Household_municipality'] = 0
data.fillna(0, inplace=True)

#data.columns.difference(test.columns) test if test and data has same columns

#print data
# print data.head()
# print data.shape
#print data.describe()

#y = data.category_num
# X = data.drop(['TV_flow_consump_cat_children', 'TV_flow_consump_cat_documentary',
#           'TV_flow_consump_cat_movies', 'TV_flow_consump_cat_leisure',
#           'TV_flow_consump_cat_art', 'TV_flow_consump_cat_music',
#           'TV_flow_consump_cat_news', 'TV_flow_consump_cat_politics',
#           'TV_flow_consump_cat_series', 'TV_flow_consump_cat_show',
#           'TV_flow_consump_cat_sports', 'TV_flow_consump_cat_other', 'category_num'], axis=1)
#X = data.drop(['category_num'], axis=1)

clf2 = joblib.load('model.pkl')

# Predict data set using loaded model
pred = clf2.predict(data)
print np.around(pred)

# Index([u'TV_flow_consump_cat_news', u'TV_flow_consump_cat_movies',
#        u'TV_flow_consump_cat_children', u'TV_flow_consump_cat_art',
#        u'TV_flow_consump_cat_show', u'TV_flow_consump_cat_sports',
#        u'TV_flow_consump_cat_series', u'TV_flow_consump_cat_documentary',
#        u'TV_flow_consump_cat_music', u'TV_flow_consump_cat_politics',
#        u'TV_flow_consump_cat_leisure', u'TV_flow_consump_cat_other'],
#       dtype='object')
