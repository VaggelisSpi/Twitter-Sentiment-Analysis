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
#     display_name: dm_kernel
#     language: python
#     name: dm_kernel
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
from operator import add
import random
from numpy import array

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
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

# import my functions
from myFunctions import *


# Read the train data

Location = r'../twitter_data/train2017.tsv'
df = pd.read_csv(Location, sep='\t', names=['ID_1', 'ID_2', 'Label', 'Text'])

# region
# use only a part of csv
# df = df[:10000]
# Preprocess the traindata
processed_list = preprocess(df)

# print data
# df
# endregion

# Read the test data

# region
testLocation = r'../twitter_data/test2017.tsv'
testDf = pd.read_csv(testLocation, sep='\t', names=['ID_1', 'ID_2', 'Label', 'Text'])

# use only a part of csv
testDf = testDf[:10000]

# Preprocess the testData
processed_list_test = preprocess(testDf)

# print data
# testDf
# endregion

# Read the correct results

# region
resultsLocation = r'../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt'
testResults = pd.read_csv(resultsLocation, sep='\t', names=['ID', 'Label'])

# use only a part of csv
testResults = testResults[:10000]

# print data
# testResults
# endregion

# # Do the classification

# Build label encoder for categories
le = preprocessing.LabelEncoder()
le.fit(df["Label"])

# Transform categories into numbers
y = le.transform(df["Label"])
print("y.shape is:")
print(y.shape)
y_test = le.transform(testResults["Label"])
print("y_test.shape is:")
print(y_test.shape)

# get processed content for list
processed_content = [item[1] for item in processed_list]
processed_content_test = [item[1] for item in processed_list_test]
# processed_content

# ### Vectorize content

# #### Vectorization using count vectorizer

# region
# Vectorize Content
# Choose one of the below

# CountVectorizer (BOW)

# count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
# X = count_vectorizer.fit_transform(processed_content)
# endregion

# #### Vectorization using TfId vectorizer

# region
# TfIdfVectorizer
# train and test vectors should have the same number of features

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(processed_content)
# endregion

# #### Vectorization using word embeddings
#
# Train the word embeddings model and save it to a file
#

# region
# Word embeddings
tokenize = lambda x: x.split()
tokenized_tweet = [tokenize(x) for x in processed_content] # tokenizing
# print(tokenized_tweet)
vec_size = 200
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=vec_size,  # desired no. of features/independent variables
            window=7,  # context window size
            min_count=5,
            sg=1,  # 1 for skip-gram model
            hs=0,
            negative=10,  # for negative sampling
            workers=2,  # no.of cores
            seed=34)

model_w2v.train(tokenized_tweet, total_examples= len(processed_content), epochs=20)

model_w2v.save("word2vec.model")
# endregion

# Load the trained word embeddings model

# region
model_w2v = Word2Vec.load("word2vec.model")

# tsne_plot(model_w2v)
# print(model_w2v.wv.vocab)
# print(model_w2v["obama"])
# endregion

# Make the vectors for the train data

# region
# tokenize = lambda x: x.split()
processed_content_vec = []
for tweet in processed_content:
    tweet_len = len(tweet)
    if tweet_len == 0:
        tweet_vec = sample_floats(-5.0, 5.0, vec_size)
        processed_content_vec.append(tweet_vec)
        continue
    tokenized_tweet = tokenize(tweet)
    if tokenized_tweet[0] in model_w2v.wv.vocab:
        tweet_vec = model_w2v.wv[tokenized_tweet[0]]
    else:
        tweet_vec = sample_floats(-5.0, 5.0, vec_size)
    for token in tokenized_tweet[1:]:
        if token in model_w2v.wv.vocab:
            tweet_vec = list(map(add, tweet_vec, model_w2v.wv[token]))
        else:
            tweet_vec = list(map(add, tweet_vec, sample_floats(-5.0, 5.0, vec_size)))
    final_tweet_vec = [i/tweet_len for i in tweet_vec]
    processed_content_vec.append(final_tweet_vec)

X = array(processed_content_vec)
# endregion

# Make the vectors for the test data

# region
processed_content_test_vec = []
print(len(processed_content_test))
for tweet in processed_content_test:
    tweet_len = len(tweet)
    if tweet_len == 0:
        tweet_vec = sample_floats(-5.0, 5.0, vec_size)
        processed_content_test_vec.append(tweet_vec)
        continue
    tokenized_tweet = tokenize(tweet)
    if tokenized_tweet[0] in model_w2v.wv.vocab:
        tweet_vec = model_w2v.wv[tokenized_tweet[0]]
    else:
        tweet_vec = sample_floats(-5.0, 5.0, vec_size)
    for token in tokenized_tweet[1:]:
        if token in model_w2v.wv.vocab:
            tweet_vec = list(map(add, tweet_vec, model_w2v.wv[token]))
        else:
            tweet_vec = list(map(add, tweet_vec, sample_floats(-5.0, 5.0, vec_size)))
    final_tweet_vec = [i/tweet_len for i in tweet_vec]
    processed_content_test_vec.append(final_tweet_vec)

X_test = array(processed_content_test_vec)
# endregion

# #### See theshapes of the data 

# region
# X = vectorizer.fit_transform(processed_content)
print("X.shape is:")
print(X.shape)

# X_test = vectorizer.transform(processed_content_test)
print("X_test.shape is:")
print(X_test.shape)
# endregion

# ### Classification using SVM classifier

# region
clf = svm.SVC(kernel='linear')

# fit train set
clf.fit(X, y)

# predict test set (here is the same as the train set)
y_pred = clf.predict(X_test)
print("y_pred.shape is:")
print(y_pred.shape)

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

# ### Classification using KNN classifier

# region
# Use KNNClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# fit train set
knn.fit(X, y)

# Predict test set (here is the same as the train set)
y_pred = knn.predict(X_test)

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
