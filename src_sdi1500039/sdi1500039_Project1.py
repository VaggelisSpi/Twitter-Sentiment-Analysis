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
# ## <center>Panayiotis Evangelliou</center>
# ## <center>Vaggelis Spithas</center>

# ___

# ### Do all the necessary imports for this notebook

# region
# for wordclouds
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
from wordcloud import WordCloud
from IPython.display import Image
from PIL import Image as imgWordcloud
import numpy as np

# for preprocessing
import re
from string import punctuation
from nltk.stem import StemmerI, RegexpStemmer, LancasterStemmer, ISRIStemmer, PorterStemmer, SnowballStemmer, RSLPStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords as nltkStopwords

# for classification
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from gensim.models import Word2Vec
import random
from operator import add
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# endregion

# ## __Data Analysis__

# - ### *Wordclouds*

# region
# read train data
trainData = pd.read_csv('../twitter_data/train2017.tsv', sep='\t+', escapechar="\\",
                        engine='python', names=['ID_1', 'ID_2', 'Label', 'Text'])

# make stop words
stopWords = ENGLISH_STOP_WORDS
# endregion

#   - #### Wordcloud for all tweets

# region
# make a text of all tweets
wholeText = ''
for tweetText in trainData['Text']:
    wholeText = wholeText + ' ' + tweetText

wc = WordCloud(width=600, height=600, background_color='white', stopwords=stopWords)

wc.generate(wholeText)
wc.to_file('wholeTextWordcloud.png')

Image('wholeTextWordcloud.png')
# endregion

# As we see in the above wordcloud there are many useless words like "http" so let's do some preprocessing.


# region
def replaceEmojis(text):
    """Turn emojis into smilePositive and smileNegative to reduce noise."""
    
    processedText = text.replace('0:-)', 'smilePositive')
    processedText = processedText.replace(':)', 'smilePositive')
    processedText = processedText.replace(':D', 'smilePositive')
    processedText = processedText.replace(':*', 'smilePositive')
    processedText = processedText.replace(':o', 'smilePositive')
    processedText = processedText.replace(':p', 'smilePositive')
    processedText = processedText.replace(';)', 'smilePositive')

    processedText = processedText.replace('>:(', 'smileNegative')
    processedText = processedText.replace(';(', 'smileNegative')
    processedText = processedText.replace('>:)', 'smileNegative')
    processedText = processedText.replace('d:<', 'smileNegative')
    processedText = processedText.replace(':(', 'smileNegative')
    processedText = processedText.replace(':|', 'smileNegative')
    processedText = processedText.replace('>:/', 'smileNegative')

    return processedText


def preprocessText(initText):
    """Preprocess the text"""
    
    # Replace these characters as we saw in the first wordcloud that are not useful in this form
    processedText = initText.replace("\\u002c", ',')
    processedText = processedText.replace("\\u2019", '\'')

    # Make everything to lower case
    processedText = processedText.lower()

    # Remove urls
    processedText = re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)'
                           r'*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?', ' ', processedText)

    # Remove hashtags
    processedText = re.sub(r"\B#\w\w+", ' ', processedText)

    # Remove references
    processedText = re.sub(r"\B@\w\w+", ' ', processedText)

    # Replace emojis with tags
    processedText = replaceEmojis(processedText)

    # Replace hahas
    processedText = re.sub('hahaha+', ' ', processedText)
    processedText = re.sub('haha+', ' ', processedText)

    # Remove any punctuation from the text
    for c in punctuation:
        processedText = processedText.replace(c, ' ')

    # Remove consecutive spaces
    processedText = re.sub(r" {2,}", ' ', processedText)    
    
    # Split to words
    tokens = word_tokenize(processedText)
    # Remove sropwords
    filtered = [w for w in tokens if w not in stopWords]

    # Concat the remaining words in a single string again
    if not filtered:  # list is empty
        processedText = ''
    else:
        processedText = filtered[0]
        for word in filtered[1:]:
            processedText = processedText + ' ' + word

    return processedText

def stemmingPreprocess(initText):
    # Split to words
    tokens = word_tokenize(initText)
    
    # Do the stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in tokens]
    
    # Concat the remaining words in a single string again
    if not stems:  # list is empty
        processedText = ''
    else:
        processedText = stems[0]
        for stem in stems[1:]:
            processedText = processedText + ' ' + stem

    return processedText
# endregion


# Below we conclude on what our stopwords are and then preprocess the text. 
# In the first wordcloud we saw many words that occur multiple times and don't seem to add anything regarding the sentiment so we will remove them. 

# region
# In the first wordcloud we see that there are many words that are not included in stop words
# So let's add our stop words
myAdditionalStopWords = ['tomorrow', 'today', 'day', 'tonight', 'sunday',
                         'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                         'saturday', 'week', 'just', 'going', 'time','say','said']
stopWords = (stopWords.union(myAdditionalStopWords)).union(nltkStopwords.words('english'))

for index, row in trainData.iterrows():
    initialText = row["Text"]
    trainData.loc[index, "Text"] = preprocessText(initialText)
# endregion

# Let's make again a wordcloud for the text of all tweets

# region
# make a text of all tweets
wholeText = ''
for tweetText in trainData['Text']:
    wholeText = wholeText + ' ' + tweetText

generalMask = np.array(imgWordcloud.open("generalMask.png"))

wc = WordCloud(background_color="white", mask=generalMask, max_words=100,
               stopwords=stopWords, contour_width=3, contour_color='steelblue')

# generate word cloud
wc.generate(wholeText)

# store to file
wc.to_file('wholeTextCleanWordcloud.png')

Image('wholeTextCleanWordcloud.png')
# endregion

# #### Make content for each category of all tweets

# For each category we make a dictionary that maps the category to a string that contains all the tweets for the respective category as one.

# region
tweetCategories = list(set(trainData['Label']))

# make a dictionary of form {category:contentString}
contentDict = {category: '' for category in tweetCategories}

# fill the content of each category
for (content, category) in zip(trainData['Text'], trainData['Label']):
    contentDict[category] = contentDict[category] + ' ' + content
# endregion

#   - #### Wordcloud for positive tweets

# region
positiveMask = np.array(imgWordcloud.open("positiveMask.png"))

wc = WordCloud(background_color="white", mask=positiveMask, max_words=100,
               stopwords=stopWords, contour_width=3, contour_color='steelblue')

# generate word cloud
wc.generate(contentDict['positive'])

# store to file
wc.to_file('positiveWordcloud.png')

Image('positiveWordcloud.png')
# endregion

#   - #### Wordcloud for negative tweets

# region
negativeMask = np.array(imgWordcloud.open("negativeMask.png"))

wc = WordCloud(background_color="white", mask=negativeMask, max_words=100,
               stopwords=stopWords, contour_width=3, contour_color='steelblue')

# generate word cloud
wc.generate(contentDict['negative'])

# store to file
wc.to_file('negativeWordcloud.png')

Image('negativeWordcloud.png')
# endregion

#   - #### Wordcloud for neutral tweets

# region
neutralMask = np.array(imgWordcloud.open("neutralMask.png"))

wc = WordCloud(background_color="white", mask=neutralMask, max_words=100,
               stopwords=stopWords, contour_width=3, contour_color='steelblue')

# generate word cloud
wc.generate(contentDict['neutral'])

# store to file
wc.to_file('neutralWordcloud.png')

Image('neutralWordcloud.png')
# endregion

# ___

# ## __Classification__

# Below we will proceed to the classification of the text. For that purpose we will use 2 classifiers, kNN and SVM, and 3 different ways of vectorization, bag-of-words, TF-IDF and word embeddings. We provide two different ways in order to produce the word embeddings, a model we trained ourselves and the use of pretrained word embeddings. Here we use only the pretrained embeddings as after experimentation we concluded that these provide better results

#   - #### Classification using SVM classifier


def SvmClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the SVM classifier of scikit-learn    
    """
    
    clf = svm.SVC(kernel='linear', C=1, probability=True)

    # fit train set
    clf.fit(trainX, trainY)
    
    # Predict test set
    predY = clf.predict(testX)

    # Classification_report
    print(classification_report(testY, predY, target_names=list(labelEncoder.classes_)))

    return accuracy_score(testY, predY)


#   - #### Classification using KNN classifier

def KnnClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the kNN classifier of scikit-learn    
    """
    knn = KNeighborsClassifier(n_neighbors=5)

    # fit train set
    knn.fit(trainX, trainY)

    # Predict test set
    predY = knn.predict(testX)

    # Classification_report
    print(classification_report(testY, predY, target_names=list(labelEncoder.classes_)))

    return accuracy_score(testY, predY)

# Prepare train and test data that we will need below


# region
# read test data
testData = pd.read_csv('../twitter_data/test2017.tsv', sep='\t', names=['ID_1', 'ID_2', 'Label', 'Text'])

# preprocess test data
for index, row in testData.iterrows():
    initialText = row["Text"]
    testData.loc[index, "Text"] = preprocessText(initialText)

# read test results
testResults = pd.read_csv('../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt',
                          sep='\t', names=['ID', 'Label'])

# Build label encoder for categories
le = preprocessing.LabelEncoder()
le.fit(trainData["Label"])

# Transform categories into numbers
trainY = le.transform(trainData["Label"])
testY = le.transform(testResults["Label"])

accuracyDict = dict()

trainNotStemmed = trainData
testNotStemmed = testData

# Let's do stemming
for index, row in trainData.iterrows():
    initialText = row["Text"]
    trainData.loc[index, "Text"] = stemmingPreprocess(initialText)

for index, row in testData.iterrows():
    initialText = row["Text"]
    testData.loc[index, "Text"] = stemmingPreprocess(initialText)
# endregion

# ## __Vectorization__

# Let's do classification using 3 different ways of vectorization

#   - #### Bag-of-words vectorization

# region
bowVectorizer = CountVectorizer(max_features=3000)

trainX = bowVectorizer.fit_transform(trainData['Text'])
testX = bowVectorizer.transform(testData['Text'])

print('-------------SVM Classification Report with BOW Vectorization-------------')
accuracyDict["BOW-SVM"] = SvmClassification(trainX, trainY, testX, testY, le)

print('-------------KNN Classification Report with BOW Vectorization-------------')
accuracyDict["BOW-KNN"] = KnnClassification(trainX, trainY, testX, testY, le)
# endregion

#   - #### Tf-idf vectorization

# region
tfIdfVectorizer = TfidfVectorizer(max_features=3000)

trainX = tfIdfVectorizer.fit_transform(trainData['Text'])
testX = tfIdfVectorizer.transform(testData['Text'])

print('-------------SVM Classification Report with TfIdf Vectorization-------------')
accuracyDict["TfIdf-SVM"] = SvmClassification(trainX, trainY, testX, testY, le)

print('-------------KNN Classification Report with TfIdf Vectorization-------------')
accuracyDict["TfIdf-KNN"] = KnnClassification(trainX, trainY, testX, testY, le)
# endregion

#   - #### Word embeddings vectorization

# Train the word embeddings model and save it. If the model is already trained and saved then you only have to load
# it as shown in the next cell

# region
# Word embeddings
tokenize = lambda x: x.split()
tokens = [word_tokenize(row["Text"]) for index, row in trainData.iterrows()]  # tokenizing

vec_size = 200
model_w2v = gensim.models.Word2Vec(
            tokens,
            size=200,  # desired no. of features/independent variables
            window=5,  # context window size
            min_count=2,
            sg=1,  # 1 for skip-gram model
            hs=0,
            negative=10,  # for negative sampling
            workers=2,  # no.of cores
            seed=34)

model_w2v.train(tokens, total_examples=trainData.shape[0], epochs=20)

model_w2v.save("word2vec.model")
# endregion

# Load the word embeddings model

model_w2v = Word2Vec.load("word2vec.model")

# Read pre-trained Word Embeddings

# region
embeddings_dict = {}

# f = open("datastories.twitter.50d.txt", "r", encoding="utf-8")
# vec_size = 50

# f = open("datastories.twitter.200d.txt", "r", encoding="utf-8")
# vec_size = 200

f = open("datastories.twitter.300d.txt", "r", encoding="utf-8")
vec_size = 300

for i, line in enumerate(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_dict[word] = coefs
# endregion

# Let's make a tsne plot for pre-trained word embeddings

# region
labels = []
tokens = []

maxWords = 200
counter = 0

for word, value in embeddings_dict.items():
    tokens.append(value)
    labels.append(word)
    counter += 1
    if(counter == maxWords):
        break

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []

for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(16, 16))
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
           textcoords='offset points',
                     ha='right',
                     va='bottom')
plt.show()
# endregion

# Use the following function to vectorize the data using the word embeddings vectorizer

# region
def sample_floats(low=-1.0, high=1.0, k=1):
    """ 
    Return a k-length list of unique random floats in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result


def wordEmbeddingsVectorizer(data):
    """
    Vectorize the data based on the model we trained ourselves.
    """
    text_vec = []
    for index, row in data.iterrows():
        text = row["Text"]
        text_len = len(text)
        if text_len == 0:
            tweet_vec = sample_floats(-5.0, 5.0, vec_size)
            text_vec.append(tweet_vec)
            continue
        tokens = word_tokenize(text)
        if tokens[0] in model_w2v.wv.vocab:
            tweet_vec = model_w2v.wv[tokens[0]]
        else:
            tweet_vec = sample_floats(-5.0, 5.0, vec_size)
        for token in tokens[1:]:
            if token in model_w2v.wv.vocab:
                tweet_vec = list(map(add, tweet_vec, model_w2v.wv[token]))
            else:
                tweet_vec = list(map(add, tweet_vec, sample_floats(-5.0, 5.0, vec_size)))
        final_tweet_vec = [i / text_len for i in tweet_vec]
        text_vec.append(final_tweet_vec)

    return np.array(text_vec)

def wordEmbeddingsPreTrainedVectorizer(data):
    """
    Vectorize the data based on a pretrained model we downloaded.
    """
    text_vec = []
    for index, row in data.iterrows():
        text = row["Text"]
        text_len = len(text)
        # If the text is empty make a random vector         
        if text_len == 0:
            tweet_vec = sample_floats(-3.0, 3.0, vec_size)
            text_vec.append(tweet_vec)
            continue
        tokens = word_tokenize(text)
        if tokens[0] in embeddings_dict:
            tweet_vec = embeddings_dict[tokens[0]]
        else:
            tweet_vec = sample_floats(-3.0, 3.0, vec_size)  # make a random vector if the word is not in the model
        for token in tokens[1:]:
            if token in embeddings_dict:
                tweet_vec = list(map(add, tweet_vec, embeddings_dict[token]))
            else:
                # make a random vector if the word is not in the model
                tweet_vec = list(map(add, tweet_vec, sample_floats(-3.0, 3.0, vec_size)))
        final_tweet_vec = [i / text_len for i in tweet_vec]
        text_vec.append(final_tweet_vec)

    return np.array(text_vec)
# endregion

# Read the lexica

def readDictionary(fileName):
    dictFile = open(fileName, "r")
    dictionary = dict()
    for line in dictFile:
        words = line.split()
        text = ' '.join(words[:-1])
        dictionary[text] = float(words[-1])

    return dictionary


# For every tweet calculate the values of each dictionary and append them as extra features in the feature vector

def getDictValues(data, vector):
    extra_feats = []
    for index, row in data.iterrows():
        text = row["Text"]
        
        affinScore = 0.0
        emotweetScore = 0.0
        genericScore = 0.0
        nrcScore = 0.0
        nrctagScore = 0.0
        
        # Empty rows are not considered strings if read from a csv.
        if not isinstance(text, str):
            l = [affinScore, emotweetScore, genericScore, nrcScore, nrctagScore]
            extra_feats.append(l)
            continue
                    
        text_len = len(text)
        # If the tweet is empty after preprocessing add zeroes         
        if text_len == 0:
            l = [affinScore, emotweetScore, genericScore, nrcScore, nrctagScore]
            extra_feats.append(l)
            continue

        tokens = word_tokenize(text)
        # If the tweet is empty after preprocessing add zeroes
        if tokens == []:
            extra_feats.append(l)
            continue

        text_len = len(tokens)

        # Calculate the sum of the values of each dict
        for token in tokens:
            if token in affinDict:
                affinScore += affinDict[token]
            if token in emotweetDict:
                emotweetScore += emotweetDict[token]
            if token in genericDict:
                genericScore += genericDict[token]
            if token in nrcDict:
                nrcScore += nrcDict[token]
            if token in nrctagDict:
                nrctagScore += nrctagDict[token]
        # Get the average values for each text
        affinScore /= text_len
        emotweetScore /= text_len
        genericScore /= text_len
        nrcScore /= text_len
        nrctagScore /= text_len
        l = [affinScore, emotweetScore, genericScore, nrcScore, nrctagScore]
        extra_feats.append(l)

    return np.append(vector, np.array(extra_feats), axis=1)

# Read the dictionary files and store them in python dictionaries

affinDict = readDictionary("../lexica/affin/affin.txt")
emotweetDict = readDictionary("../lexica/emotweet/valence_tweet.txt")
genericDict = readDictionary("../lexica/generic/generic.txt")
nrcDict = readDictionary("../lexica/nrc/val.txt")
nrctagDict = readDictionary("../lexica/nrctag/val.txt")

# Vectorize the content using word embeddings vectorizer. Then add some extra features using the dictionary files

# region
# these are for our trained word embeddings
# trainX = wordEmbeddingsVectorizer(trainData)
# testX = wordEmbeddingsVectorizer(testData)

# these are for pre-trained wordEmbeddings
trainX = wordEmbeddingsPreTrainedVectorizer(trainNotStemmed)
testX = wordEmbeddingsPreTrainedVectorizer(testNotStemmed)

print('-------------SVM Classification Report with Word Embeddings Vectorization without lexica-------------')
accuracyDict["WordEmbed-SVM-without-lexica"] = SvmClassification(trainX, trainY, testX, testY, le)

print('-------------KNN Classification Report with Word Embeddings Vectorization without lexica-------------')
accuracyDict["WordEmbed-KNN-without-lexica"] = KnnClassification(trainX, trainY, testX, testY, le)
# endregion

# Let's try word embeddings with lexica

# region
trainX = getDictValues(trainNotStemmed, trainX)
testX = getDictValues(testNotStemmed, testX)

print('-------------SVM Classification Report with Word Embeddings Vectorization with lexica-------------')
accuracyDict["WordEmbed-SVM-with-lexica"] = SvmClassification(trainX, trainY, testX, testY, le)

print('-------------KNN Classification Report with Word Embeddings Vectorization with lexica-------------')
accuracyDict["WordEmbed-KNN-with-lexica"] = KnnClassification(trainX, trainY, testX, testY, le)
# endregion

# ## __Final Results__

# region
resultsData = {r'Vectorizer \ Classifier': ['BOW', 'Tfidf', 'Word Embeddings without lexica',
                                            'Word Embeddings with lexica'],
               'KNN': [accuracyDict["BOW-KNN"], accuracyDict["TfIdf-KNN"], accuracyDict["WordEmbed-KNN-without-lexica"],
                       accuracyDict["WordEmbed-KNN-with-lexica"]],
               'SVM': [accuracyDict["BOW-SVM"], accuracyDict["TfIdf-SVM"], accuracyDict["WordEmbed-SVM-without-lexica"],
                       accuracyDict["WordEmbed-KNN-with-lexica"]]}

resultsDataFrame = pd.DataFrame(data=resultsData)

resultsDataFrame
# endregion

# **Comments and remarks**
# - After various experimentations regarding the max_features parameter of bow and tf-idf we concluded that for the value 3000, the results are similar or better than other values we tried or the default value, both for the SVM and kNN classifiers.
# - We observe that the SVM classifier performs better than kNN
# - Moreover, after experimentations on preproccessing we noticed that if we stemmed our data we have better accuracy for both BOW and TF-IDF opposing to not doing stemming
# - In vectorization with word embeddings we decided not to use stemming because we used the lexica, in whihc the words are in their normal form
# - We tried making our own word embeddings model as well as a pretrained one. Eventually we saw that the preptrained model has better results and as such we used this.
# - We observed that the minimun value in the pretrained word embeddings model with 200 dimensions is -3.224762 and the maximum is 5.65153, so we decided to create our random vectors with values between -3 amd 3
# - In order to rerun the notebook from the beggining you will have to change the file paths on your own if you plan to use different files
# - Keep in mind that when reruning from the beggining it miht take some time for the vectorizations to finish
#
