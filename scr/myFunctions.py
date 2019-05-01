# Import all libraries needed for the tutorial

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

def preprocess_data(df):
    '''
    Preprocess the data and calculate some stats.
    '''
    ret = []
    for index, row in df.iterrows():
        # print(row['Text'])
        # Retrieve he text and make it lower case
        text = row['Text'].lower()
        # print(text)
        # remove all punctuation
        text = text.replace(punctuation, '')
        print(text)
        # Split the text in words
        tokens = word_tokenize(text)
        # print(tokens)
        word_count = len(tokens)
        print("word_count =", word_count)
        # Count the positive, negative and neutral words as well as the
        # sentiment score
        positive_words_count = 0
        negative_words_count = 0
        affin_cost = 0
        generic_cost = 0
        for word in tokens:
            # calculate seniment value for the affinity dictionary
            with open('../lexica/affin/affin.txt') as csvfile:
                readCSV = csv.reader(csvfile, delimiter='\t')
                for row in readCSV:
                    # print(row[0])
                    if word == row[0]:
                        # print("Found word", row[0], " with cost", row[1])
                        affin_cost += float(row[1])

            # calculate seniment value for the generic dictionary
            with open('../lexica/generic/generic.txt') as csvfile:
                readCSV = csv.reader(csvfile, delimiter='\t')
                for row in readCSV:
                    # print(row[0])
                    if word == row[0]:
                        # print("Found word", row[0], " with cost", row[1])
                        generic_cost += float(row[1])

        ret += [(word_count, generic_cost, affin_cost)]

    return ret


def correct_words(spellchecker, words, add_to_dict=[]):
    enc = spellchecker.get_dic_encoding()   # get the encoding for later use in decode()

    # add custom words to the dictionary
    for w in add_to_dict:
        spellchecker.add(w)

    # auto-correct words
    corrected = []
    for w in words:
        ok = spellchecker.spell(w)   # check spelling
        if not ok:
            suggestions = spellchecker.suggest(w)
            if len(suggestions) > 0:  # there are suggestions
                best = suggestions[0].decode(enc)   # best suggestions (decoded to str)
                corrected.append(best)
            else:
                corrected.append(w)  # there's no suggestion for a correct word
        else:
            corrected.append(w)   # this word is correct

    return corrected


def spellcheck(text):
    # spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_EN.dic',
    #                                 '/usr/share/hunspell/en_EN.aff')
    # words =
    # correct_words(spellchecker, )
    # chkr = enchant.checker.SpellChecker("en_EN")
    # chkr.set_text(text)
    # for err in chkr:
    #     print(err.word)
    #     sug = err.suggest()[0]
    #     err.replace(sug)
    #
    # c = chkr.get_text()#returns corrected text
    # return c
    pass


def replace_emojis(text):
    '''
    Turn emojis into smile_positive and smile_negative to reduce noise
    :param text:
    :return:
    '''
    processed_text = text.replace('0:-)', 'smile_positive')
    processed_text = processed_text.replace(':)', 'smile_positive')
    processed_text = processed_text.replace(':D', 'smile_positive')
    processed_text = processed_text.replace(':*', 'smile_positive')
    processed_text = processed_text.replace(':o', 'smile_positive')
    processed_text = processed_text.replace(':p', 'smile_positive')
    processed_text = processed_text.replace(';)', 'smile_positive')

    processed_text = processed_text.replace('>:(', 'smile_negative')
    processed_text = processed_text.replace(';(', 'smile_negative')
    processed_text = processed_text.replace('>:)', 'smile_negative')
    processed_text = processed_text.replace('d:<', 'smile_negative')
    processed_text = processed_text.replace(':(', 'smile_negative')
    processed_text = processed_text.replace(':|', 'smile_negative')
    processed_text = processed_text.replace('>:/', 'smile_negative')

    return processed_text


def preprocess(df):
    """
    Process the data

    :param df:
    :return:
    """

    ret = []
    for index, row in df.iterrows():
        text = row['Text']
        # print(text)

        # Make everything to lower case
        processed_text = text.lower()

        # Remove urls
        processed_text = re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)'
                                r'*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?', '', processed_text)

        # Remove hashtags
        processed_text = re.sub(r"\B#\w\w+", '', processed_text)

        # Remove tags
        processed_text = re.sub(r"\B@\w\w+", '', processed_text)

        # Replace emojis with tags
        processed_text = replace_emojis(processed_text)

        # Replace hahas
        processed_text = re.sub('hahaha+', '', processed_text)
        processed_text = re.sub('haha+', '', processed_text)

        # Remove consecutive spaces
        processed_text = re.sub(r" {2,}", ' ', processed_text)

        # Remove any punctuation from the text
        for c in punctuation:
            processed_text = processed_text.replace(c, '')

        # Split to words
        tokens = word_tokenize(processed_text)

        stemmer = PorterStemmer()
        stems = [stemmer.stem(token) for token in tokens]

        filtered = [w for w in stems if w not in stopwords.words('english')]

        processed_text = filtered[0]
        for token in filtered[1:]:
            processed_text = processed_text + ' ' + token

        # print(processed_text)
        ret += [(index, processed_text)]
        # print(df.loc[[index]])
        # break
    return ret