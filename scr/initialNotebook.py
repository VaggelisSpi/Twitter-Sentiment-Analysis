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

# # Initial Jupyter notebook for data mining project
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

# region
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
# endregion

# import my functions
from myFunctions import *

# Read the data
Location = r'../twitter_data/myTrain.tsv'
df = pd.read_csv(Location, sep='\t', names=['ID_1', 'ID_2', 'Label', 'Text'])

# print data
df

# Preprocess the data
processed_list = preprocess(df)

# print processed data
processed_list

# # Do the classification

# Build label encoder for categories
le = preprocessing.LabelEncoder()
le.fit(df["Label"])

# Transform categories into numbers
y = le.transform(df["Label"])

# get processed content for list
processed_content = [item[1] for item in processed_list]
processed_content

# region
# Vectorize Content
# Choose one of the below

# CountVectorizer (BOW)

# count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
# X = count_vectorizer.fit_transform(processed_content)

# TfIdfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_content)

# endregion

# Classification using SVM classifier

# region
clf = svm.SVC(kernel='linear')

# fit train set
clf.fit(X, y)

# predict test set (here is the same as the train set)
y_pred = clf.predict(X)

print('\npredictions of test set (which is the same as the train set) are:')
print(y_pred)

# Transform predictions to text
predicted_categories = le.inverse_transform(y_pred)
print('\npredictions of test set in text form are:')
print(predicted_categories)

# Classification_report
print('\nclassification report for these predictions is:')
print(classification_report(y, y_pred, target_names=list(le.classes_)))
# endregion

# Classification using KNN classifier

# region

# Use KNNClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# fit train set
knn.fit(X, y)

# Predict test set (here is the same as the train set)
y_pred = knn.predict(X)

print('\npredictions of test set (which is the same as the train set) are:')
print(y_pred)

# Transform predictions to text
predicted_categories = le.inverse_transform(y_pred)
print('\npredictions of test set in text form are:')
print(predicted_categories)

# Classification_report
print('\nclassification report for these predictions is:')
print(classification_report(y, y_pred, target_names=list(le.classes_)))

# endregion
