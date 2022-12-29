import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np
import pandas as pd
import dask.dataframe as dd

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.metrics import accuracy_score,  classification_report, f1_score, roc_auc_score


df = pd.read_csv(r'C:\Users\ania0\Desktop\repozytoria\pracInz\gotowe_bazy_danych\Tweets.csv')


# df=df.replace(to_replace="positive",value="1")
# df=df.replace(to_replace="neutral",value="0")
# df=df.replace(to_replace="negative",value="-1")

y = df['airline_sentiment']

vectorizer = CountVectorizer(stop_words='english')
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

X = tfidf_vectorizer.fit_transform(df['text'])
X = vectorizer.fit_transform(df['text'])


X_train, X_test, y_train, y_test=train_test_split(X,y,stratify=y, test_size=0.3, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
accuracy = accuracy_score(predicted, y_test)
print(accuracy)


# airlane_review = np.array(['Number of samples encountered for each during fitting'])
# airlane_review_vector = vectorizer.transform(airlane_review)
# print(model.predict(airlane_review_vector))


df = pd.read_csv(r'C:\Users\ania0\Desktop\repozytoria\pracInz\bazy_danych\ ' + 'dataset_of_tweets.csv')
list_of_tweets = np.array(list(df['Tekst'])) 
tweet_vector = vectorizer.transform(list_of_tweets)
predict_ = model.predict(tweet_vector)

result = np.column_stack((predict_, list_of_tweets))
column = ['Wydźwięk', 'Tekst']
df = pd.DataFrame(result, columns=column)
df.to_csv(r'C:\Users\ania0\Desktop\repozytoria\pracInz\bazy_danych\ ' + 'wyniki.csv')










