import re
import numpy as np

import nltk
nltk.download('omw-1.4')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from collections import Counter


SYMBOLS = ' {}()[].,:;+-*/&$...|<>=~^!?“’"'

def joinList(word_lists):
    all_words_list = []
    for sublist in word_lists:
        for item in sublist:
            all_words_list.append(item)
    return all_words_list

def tokenizer(text):
    tokenizer=TweetTokenizer() #tokenizacja tweeta - podzielenie zdania na części
    text_tokenizer = tokenizer.tokenize(text.lower())
    return text_tokenizer

def stopWords(text):
    stop_words = set(stopwords.words('english')) #niepotrzebne słowa w języku angielskim
    text_stopwords = [w for w in text if not w.lower() in stop_words] #usunięcie stop words
    return text_stopwords

def filtrationSYM(text):
    m = []
    for element in text: #usunięcie niepotrzebnych symboli
        temp = ''
        for ch in element:
            if ch not in SYMBOLS and ch.isalpha():
                temp += ch
        m.append(temp)
    return m

def deleteLink(text):
    return re.sub(r'http\S+', '', text)

def lemmatizer(text):
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text]
    return text

def stemmer(text):
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return text


def top20words(data):
    count_words = Counter(data)
    top20 = Counter(data).most_common(20)
    return top20


   

