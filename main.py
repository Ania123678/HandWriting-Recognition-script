import functions as fun
import Bayes
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def main():

    ##---------------   Wykres 20 najbardziej użytych słów    ---------------##
    df = pd.read_csv(r'C:\Users\ania0\Desktop\repozytoria\pracInz\bazy_danych\ ' + 'dataset_of_tweets.csv')
    
    all_words = df['Słowa kluczowe'].tolist()
    q = []
    for m in all_words:
        tmp = []
        list_of_symbols = ['[', "'", ']']
        for s in list_of_symbols:
            m = m.replace(s,'')
        tmp = m.split(', ')
        tmp = list(filter(None,tmp))
        q.append(tmp)

    all_words_list = fun.joinList(q)
    top20 = fun.top20words(all_words_list)
    print(top20)
    df2 = pd.DataFrame(top20, columns=['słowo', 'ilość'])

    osie = df2.plot.bar(x='słowo', y='ilość',legend=False)
    osie.set_ylabel('')
    osie.set_xlabel('')
    osie.set_title('20 najczęściej pojawiających się słów w tweetach')
    plt.show()
    print('wykres')

    ##---------------   Klasyfikacja i wykres    ---------------##
    wyniki = Bayes.predict_
    count_wyniki =  Counter(wyniki)
    count_wyniki = Counter(count_wyniki).most_common(3)
    print(count_wyniki)


    df3 = pd.DataFrame(count_wyniki, columns=['opinia', 'ilość'])
    osie = df3.plot.bar(x='opinia', y='ilość',legend=False)
    osie.set_ylabel('')
    osie.set_xlabel('')
    osie.set_title('Opinia użytkowników')
    plt.show()
    
    print('koniec')

if __name__ == "__main__":
    main()

