import tweepy
import verf_keys
import functions as fun
import pandas as pd
 
 ##---------------   Autoryzacja i stworzenie zapytania    ---------------##
client = tweepy.Client(bearer_token=verf_keys.Bearer_Token)
api = tweepy.API(client)
#query = 'flu -is:retweet lang:EN'
query = 'flu lang:EN'
start = '2022-12-22T12:00:00Z'
end = '2022-12-29T11:18:00Z'
response = client.search_recent_tweets(query=query, start_time = start, end_time = end ,max_results = 100, tweet_fields=['created_at', 'public_metrics', 'lang'], user_fields = ['location'], expansions=['author_id'])
#---------------   Stworzenie pliku csv    ---------------##
users = {u['id']: u for u in response.includes['users']}
column = ['Autor','Data','Lokalizacja', 'Retweet', 'Polubienia','Ilość odpowiedzi', 'Tekst','Słowa kluczowe']
data = []
all_words = []
for tweet in response.data:
    #i = 0
    if users[tweet.author_id]:
        user = users[tweet.author_id]
        tweet_retweet = tweet.public_metrics['retweet_count']
        tweet_reply_count = tweet.public_metrics['reply_count']
        tweet_likes = tweet.public_metrics['like_count']
        
        user_location = user.location
        tweet_text = tweet.text
        dL = fun.deleteLink(tweet_text)
        t = fun.tokenizer(dL)
        sW = fun.stopWords(t)
        dSym = fun.filtrationSYM(sW)
        lem = fun.lemmatizer(dSym)
        stem = fun.stemmer(lem)
        lem = list(filter(None,stem))
        all_words.append(lem)
        data.append([user.username, tweet.created_at, user_location ,tweet_retweet, tweet_likes, tweet_reply_count ,tweet.text, lem])
        #i = i + 1
df = pd.DataFrame(data, columns=column)
#df = df.drop_duplicates() ??
df.to_csv(r'C:\Users\ania0\Desktop\repozytoria\pracInz\bazy_danych\ ' + 'dataset_of_tweets.csv')