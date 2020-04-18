# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:30:57 2020

@author: John
"""

import nltk
#dir(nltk)
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
#pd.set_option('display.max_colwidth',100)
pd.set_option('display.max_columns', None) 

stopwords.words('english')[0:500:25]

# Read in the raw text
rawData = open("SMSSpamCollection.tsv").read()

# Print the raw data
rawData[0:500]

parsedData = rawData.replace('\t', '\n').split('\n')
parsedData[0:5]

labelList =parsedData[0::2] ##start:end:step
textList = parsedData[1::2]

print(labelList[0:5])
print(textList[0:5])

print(labelList[-5:])

fullCorpus = pd.DataFrame({'label': labelList[:-1], 'body_list': textList})
fullCorpus.head()
fullCorpus.tail()

dataSet = pd.read_csv("SMSSpamCollection.tsv",sep='\t',header=None)
dataSet.head()
dataSet.tail()
print(len(dataSet))
print(len(fullCorpus))
dataSet.columns=['label', 'body_text']
dataSet.head()

#Explore the data
# What is the shape of the dataset?

print("Input data has {} rows and {} columns".format(len(dataSet), len(dataSet.columns)))

# How many spam/ham are there?
print("Out of {} rows, {} are spam, {} are ham".format(len(dataSet),
                                                       len(dataSet[dataSet['label']=='spam']),
                                                       len(dataSet[dataSet['label']=='ham'])))
# How much missing data is there?
print("Number of null label: {}".format(dataSet['label'].isnull().sum()))
print("Number of null text: {}".format(dataSet['body_text'].isnull().sum()))
print("Number of null df: {}".format(dataSet.isnull().sum()))

##Using regular expressions in Python
re_test = 'This is a made up string to test 2 different regex methods'
re_test_messy = 'This      is a made up     string to test 2    different regex methods'
re_test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""different~regex-methods'

re.split('\s', re_test)
re.split('\s', re_test_messy)
re.split('\s+', re_test_messy)
re.split('\s+', re_test_messy1)
re.split('\W+', re_test_messy1)
re.findall('\S+', re_test)
re.findall('\S+', re_test_messy)
re.findall('\S+', re_test_messy1)
re.findall('\w+', re_test_messy)

##Replacing a specific string
pep8_test = 'I try to follow PEP8 guidelines'
pep7_test = 'I try to follow PEP7 guidelines'
peep8_test = 'I try to follow PEEP8 guidelines'

re.findall('[a-z]+', pep8_test)
re.findall('[A-Z]+', pep8_test)
re.findall('[A-Z]+', pep7_test)

re.findall('[A-Z]+[0-9]+', peep8_test)
re.sub('[A-Z]+[0-9]', 'PEP8 Python Styleguide',peep8_test)

##pre-processing text data
data = pd.read_csv("SMSSpamCollection.tsv",sep='\t',header=None)
data.columns=['label', 'body_text']
data.head()

#Remove punctuation
#import string
string.punctuation

"I like NLP." == "I like NLP"
##list comprehnension for removing punctuation
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct
data["body_text_clean"] = data["body_text"].apply(lambda x: remove_punct(x))

data.head()

'NLP'=='nlp'

##Tokenization
import re
def tokenize (text):
    #tokens =  re.split('\W+', text)
    tokens = re.findall('\w+', text)
    return tokens

data["body_text_tokenized"] = data["body_text_clean"].apply(lambda x: tokenize(x.lower()))
data.head()

##remove stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text =[word for word in tokenized_list if word not in stopword]
    return text
data["body_text_nostop"] = data["body_text_tokenized"].apply(lambda x: remove_stopwords(x))
data.head()    

#Supplemental Data Cleaning: Using Stemming
ps = nltk.PorterStemmer()
dir(ps)

print(ps.stem('grows'))
print(ps.stem('growing'))
print(ps.stem('grow'))

print(ps.stem('run'))
print(ps.stem('running'))
print(ps.stem('runner'))
#raw data
import pandas as pd
import re
import string
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

data.head()

#clean up text
def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text

data['body_text_nostop'] = data['body_text'].apply(lambda x: clean_text(x.lower()))

data.head()

##stem text
def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

data['body_text_stemmed'] = data['body_text_nostop'].apply(lambda x: stemming(x))

data.head()

#Supplemental Data Cleaning: Using a Lemmatizer
import nltk

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

dir(wn)

print(ps.stem('meanness'))
print(ps.stem('meaning'))

print(wn.lemmatize('meanness'))
print(wn.lemmatize('meaning'))

print(ps.stem('goose'))
print(ps.stem('geese'))

print(wn.lemmatize('goose'))
print(wn.lemmatize('geese'))

#Lemmatize text

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))

data.head(5)

#Vectorizing Raw Data: Count Vectorization
#Read in raw data
import pandas as pd
import re
import string
import nltk
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

#Create function to remove punctuation, tokenize, remove stopwords, and stem

def clean_text2(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
##Apply CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer=clean_text2)
X_counts = count_vect.fit_transform(data['body_text'])
print(X_counts.shape)
print(count_vect.get_feature_names())

#Apply CountVectorizer to smaller sample
data_sample = data[0:20]

count_vect_sample = CountVectorizer(analyzer=clean_text2)
X_counts_sample = count_vect_sample.fit_transform(data_sample['body_text'])
print(X_counts_sample.shape)
print(count_vect_sample.get_feature_names())
# =============================================================================
# Vectorizers output sparse matrices
# Sparse Matrix: A matrix in which most entries are 0. In the interest of
#  efficient storage, a sparse matrix will be stored by only storing the 
# locations of the non-zero elements.
# =============================================================================
X_counts_sample

X_counts_df = pd.DataFrame(X_counts_sample.toarray())
X_counts_df
#Vectorizing Raw Data: N-Grams
# Creates a document-term matrix where counts still 
# occupy the cell but instead of the columns representing single terms, 
# they represent all combinations of adjacent words of length n in your text.

##read raw text

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']
#Create function to remove punctuation, tokenize, remove stopwords, and stem
def clean_text3(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text

data['cleaned_text'] = data['body_text'].apply(lambda x: clean_text3(x))
data.head()

#Apply CountVectorizer (w/ N-Grams)
from sklearn.feature_extraction.text import CountVectorizer

ngram_vect = CountVectorizer(ngram_range=(2,2))
X_counts = ngram_vect.fit_transform(data['cleaned_text'])
print(X_counts.shape)
print(ngram_vect.get_feature_names())

#Apply CountVectorizer (w/ N-Grams) to smaller sample
data_sample = data[0:20]

ngram_vect_sample = CountVectorizer(ngram_range=(2,2))
X_counts_sample = ngram_vect_sample.fit_transform(data_sample['cleaned_text'])
print(X_counts_sample.shape)
print(ngram_vect_sample.get_feature_names())

X_counts_df = pd.DataFrame(X_counts_sample.toarray())
X_counts_df.columns = ngram_vect_sample.get_feature_names()
X_counts_df

#Vectorizing Raw Data: TF-IDF
import pandas as pd
import re
import string
import nltk
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

#Create function to remove punctuation, tokenize, remove stopwords, and stem

def clean_text4(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
#Apply TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text4)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())

#Apply TfidfVectorizer to smaller sample
data_sample = data[0:20]

tfidf_vect_sample = TfidfVectorizer(analyzer=clean_text4)
X_tfidf_sample = tfidf_vect_sample.fit_transform(data_sample['body_text'])
print(X_tfidf_sample.shape)
print(tfidf_vect_sample.get_feature_names())
#Vectorizers output sparse matrices
X_tfidf_df = pd.DataFrame(X_tfidf_sample.toarray())
X_tfidf_df.columns = tfidf_vect_sample.get_feature_names()
X_tfidf_df

#Feature Engineering: Feature Creation
#read text
data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

data.head()
#Create feature for text message length
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))

data.head()
#Create feature for % of text that is punctuation

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

data.head()

##Evaluate created features
bins = np.linspace(0, 200, 40)

plt.hist(data[data['label']=='spam']['body_len'], bins, alpha=0.5, normed=True, label='spam')
plt.hist(data[data['label']=='ham']['body_len'], bins, alpha=0.5, normed=True, label='ham')
plt.legend(loc='upper left')
plt.show()

bins = np.linspace(0, 50, 40)

plt.hist(data[data['label']=='spam']['punct%'], bins, alpha=0.5, normed=True, label='spam')
plt.hist(data[data['label']=='ham']['punct%'], bins, alpha=0.5, normed=True, label='ham')
plt.legend(loc='upper right')
plt.show()

#Transformations
bins = np.linspace(0, 200, 40)

plt.hist(data['body_len'], bins)
plt.title("Body Length Distribution")
plt.show()

bins = np.linspace(0, 50, 40)

plt.hist(data['punct%'], bins)
plt.title("Punctuation % Distribution")
plt.show()

##Transform the punctuation % feature
#Box-Cox Power Transformation
for i in [1,2,3,4,5]:
    plt.hist(data['punct%']**(1/i), bins=40)
    plt.title("Transformation: 1/{}".format(str(i)))
    plt.show()
