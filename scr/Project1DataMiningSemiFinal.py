# -*- coding: utf-8 -*-
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

# # <center>Data Mining Project 1 Spring semester 2018-2019</center>
# ## <center>Παναγιώτης Ευαγγελίου &emsp; 1115201500039</center>
# ## <center>Ευάγγελος Σπίθας &emsp;&emsp;&emsp;&ensp; 1115201500147</center>

# ___

# ### Do all the necessary imports for this notebook

# region

# for wordclouds
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
from wordcloud import WordCloud
from IPython.display import Image

# endregion

# ## __Data Analysis__

# - ### *Wordclouds*

# region

# read train data
trainData = pd.read_csv('../twitter_data/train2017.tsv', sep='\t', names=['ID_1', 'ID_2', 'Label', 'Text'])

# make stop words
myAdditionalStopWords = [ ] #['said','say','just','it','says','It']
stopWords = ENGLISH_STOP_WORDS.union(myAdditionalStopWords)

# initiliaze wordcloud
wc = WordCloud(width=800,height=800,background_color = 'white', stopwords = stopWords)

trainData # printToBeRemoved

# endregion

#   - #### Wordcloud for all tweets

# region

# make a text of all tweets
wholeText = ''
for tweetText in trainData['Text']:
    wholeText = wholeText + tweetText

# wholeText # printToBeRemoved

wc.generate(wholeText)
wc.to_file('wholeTextWordcloud.png')

Image('wholeTextWordcloud.png')

# endregion

# #### make content for each category of all tweets

# region

tweetCategories = list(set(trainData['Label']))

# make a dictionary of form {category:contentString}
contentDict = { category:'' for category in tweetCategories }

# fill the content of each category
for (content, category) in zip(trainData['Text'], trainData['Label']):
    contentDict[category] = contentDict[category] + ' ' + content

# contentDict # printToBeRemoved

# endregion

#   - #### Wordcloud for positive tweets

# region

wc.generate(contentDict['positive'])
wc.to_file('positiveWordcloud.png')

Image('positiveWordcloud.png')

# endregion

#   - #### Wordcloud for negative tweets

# region

wc.generate(contentDict['negative'])
wc.to_file('negativeWordcloud.png')

Image('negativeWordcloud.png')

# endregion

#   - #### Wordcloud for neutral tweets

# region

wc.generate(contentDict['neutral'])
wc.to_file('neutralWordcloud.png')

Image('neutralWordcloud.png')

# endregion
