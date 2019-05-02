# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Initial Jupyter notebook for data mining project with real data
#

# Import all libraries needed for the tutorial

# region
# from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd  # this is how I usually import pandas
import sys  # only needed to determine Python version number
from string import punctuation
import re
import nltk
from nltk.stem import StemmerI, RegexpStemmer, LancasterStemmer, ISRIStemmer, PorterStemmer, SnowballStemmer, RSLPStemmer
from nltk import word_tokenize
# nltk.download(u'stopwords')
from nltk.corpus import stopwords

# import enchant
# import hunspell

# Enable inline plotting
# %matplotlib inline
# endregion

# For classification
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

# import my functions
from myFunctions import *


# Read the train data

# region

Location = r'../twitter_data/train2017.tsv'
df = pd.read_csv(Location, sep='\t', names=['ID_1', 'ID_2', 'Label', 'Text'])

# use only a part of csv
df = df[:10000]

# Preprocess the traindata
processed_list = preprocess(df)

# print data
df

# endregion

# read the test data

# region

testLocation = r'../twitter_data/test2017.tsv'
testDf = pd.read_csv(testLocation, sep='\t', names=['ID_1', 'ID_2', 'Label', 'Text'])

# use only a part of csv
testDf = testDf[:10000]

# Preprocess the testData
processed_list_test = preprocess(testDf)

# print data
testDf

# endregion

# read the correct results

# region

resultsLocation = r'../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt'
testResults = pd.read_csv(resultsLocation, sep='\t', names=['ID', 'Label'])

# use only a part of csv
testResults = testResults[:10000]

# print data
testResults

# endregion

# # Do the classification

# Build label encoder for categories
le = preprocessing.LabelEncoder()
le.fit(df["Label"])

# Transform categories into numbers
y = le.transform(df["Label"])
y_test = le.transform(testResults["Label"])

# get processed content for list
processed_content = [item[1] for item in processed_list]
processed_content_test = [item[1] for item in processed_list_test]
# processed_content

# region
# Vectorize Content
# Choose one of the below

# CountVectorizer (BOW)

# count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
# X = count_vectorizer.fit_transform(processed_content)

# TfIdfVectorizer

#train and test vectors should have the same number of features

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(processed_content)
print("X.shape is:")
print(X.shape)

vectorizerTest = TfidfVectorizer(max_features=10000)
X_test = vectorizerTest.fit_transform(processed_content_test)
print("X_test.shape is:")
print(X_test.shape)
# endregion

# Classification using SVM classifier

# region
clf = svm.SVC(kernel='linear')

# fit train set
clf.fit(X, y)

# predict test set (here is the same as the train set)
y_pred = clf.predict(X_test)

# print('\npredictions of test set (which is the same as the train set) are:')
# print(y_pred)

# Transform predictions to text
predicted_categories = le.inverse_transform(y_pred)
# print('\npredictions of test set in text form are:')
# print(predicted_categories)

# Classification_report
print('\nclassification report for these predictions is:')
print(classification_report(y_test, y_pred, target_names=list(le.classes_)))
# endregion

# Classification using KNN classifier

# region

# Use KNNClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# fit train set
knn.fit(X, y)

# Predict test set (here is the same as the train set)
y_pred = knn.predict(X)

# print('\npredictions of test set (which is the same as the train set) are:')
# print(y_pred)

# Transform predictions to text
predicted_categories = le.inverse_transform(y_pred)
# print('\npredictions of test set in text form are:')
# print(predicted_categories)

# Classification_report
print('\nclassification report for these predictions is:')
print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

# endregion
