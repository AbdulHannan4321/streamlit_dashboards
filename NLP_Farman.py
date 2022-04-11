import streamlit as st
from google_play_scraper import app
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib_inline
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
lr = LogisticRegression()



st.markdown('''# __Sentiment Analysis of Mobile Banking App__

This application is developed by __Abdul Hannan__''')


from google_play_scraper import Sort, reviews_all

def app_review(x):
    # pak_review = reviews_all("app.com.brd", sleep_milliseconds=0,lang="en",country="pk",sort = Sort.MOST_RELEVANT)
    pak_review = reviews_all(x, sleep_milliseconds=0,lang="en",country="pk",sort = Sort.MOST_RELEVANT)
    df = pd.DataFrame(np.array(pak_review), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    # st.write(df)
    
    # preprocessing
    def remove_punctuation(text):
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree

    df['clean_msg']= df['content'].apply(lambda x:remove_punctuation(x))
    df['msg_lower']= df['clean_msg'].apply(lambda x: x.lower())
    df['word_digits']= df['msg_lower'].apply(lambda x: re.sub('W*dw*','',x))
    df['non_english']=[words.replace("Ã", "") for words in df['msg_lower']]
    df['white_spaces']= df['non_english'].apply(lambda x: re.sub(' +',' ',x))
    def tokenization(text):
        tokens = re.split('W+',text)
        return tokens
#applying function to the column
    df['msg_tokenied']= df['white_spaces'].apply(lambda x: tokenization(x))
    stopwords = nltk.corpus.stopwords.words('english')
    # stopwords
    def remove_stopwords(text):
        output= [i for i in text if i not in stopwords]
        return output
    df['no_stopwords']= df['msg_tokenied'].apply(lambda x:remove_stopwords(x))
    df['single_character'] = df['no_stopwords'].astype(str).str.replace(r'\b\w\b','').str.replace(r'\s+',' ')
    porter_stemmer = PorterStemmer()
    def stemming(text):
        stem_text = [porter_stemmer.stem(word) for word in text]
        return stem_text
    df['msg_stemmed']=df['non_english'].apply(lambda x: stemming(x))
    # df = df[df['score'] != 3]
    df['sentiment'] = df['score'].apply(lambda rating : +1 if rating > 3 else -1)

    # Stopwords
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    textt = " ".join(review for review in df.content)
    wordcloud = WordCloud(stopwords=stopwords).generate(textt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    sw_plot = plt.show()
    # plt.savefig('wordcloud11.png')

    positive = df[df['sentiment'] == 1]
    negative = df[df['sentiment'] == -1]

    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href","good","great"]) 
## good and great removed because they were included in negative sentiment
    # positive stopwords
    pos = " ".join(review for review in positive.non_english)
    wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud12.png')
    p_plot = plt.show()

    #negative stopwords
    neg = " ".join(review for review in negative.non_english)
    wordcloud2 = WordCloud(stopwords=stopwords).generate(neg)
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud13.png')
    p_plot = plt.show()


    #Application Score
    fig = px.histogram(df, x="score")
    fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=0)
    fig.update_layout(width=800,height=500,title_text='Application Score')


    # Negative Postive Review
    df['sentimentt'] = df['sentiment'].replace({-1 : 'negative'})
    df['sentimentt'] = df['sentimentt'].replace({1 : 'positive'})
    fig_2 = px.histogram(df, x="sentimentt")
    fig_2.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
    fig_2.update_layout(width=800,height=500,title_text='Application Sentiment')
    index = df.index
    df['random_number'] = np.random.randn(len(index))
    
    #Training and Testing
    train = df[df['random_number'] <= 0.8]
    test = df[df['random_number'] > 0.8]
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train['single_character'])
    test_matrix = vectorizer.transform(test['single_character'])
    X_train = train_matrix
    X_test = test_matrix
    y_train = train['sentiment']
    y_test = test['sentiment']
    lr.fit(X_train,y_train)
    predictions = lr.predict(X_test)

    new = np.asarray(y_test)
    cm=confusion_matrix(predictions,y_test)
    cmr =classification_report(predictions,y_test)
    st.write(df)
    st.write(fig)
    # st.write(sw_plot)
    # st.write(p_plot)
    st.write(fig_2)
    st.write(cm)
    st.write(cmr)
    
bank_dict = {"UBL": "app.com.brd", "ABL": "com.ofss.digx.mobile.android.allied",
             "AlBaraka":"pk.com.albaraka.mobileapp","Alfalah":"com.base.bankalfalah","Askari":"com.askari", "Bank Al-Habib":"com.ofss.digx.mobile.obdx.bahl",
             "Bank of Khyber":"com.temenos.bok","Bank of Punjab":"com.paysys.nbpdigital","Dubai Islamic Bank":"com.avanza.ambitwizdib","EasyPaisa":"pk.com.telenor.phoenix",
             "Faysal Bank":"com.paysys.nbpdigital","First Women Bank":"com.avanza.digitalfwbl","HBL": "com.hbl.android.hblmobilebanking",
             "Habib Metropolitian Bank":"com.avanza.ambitwizhmb","JazzCash":"com.techlogix.mobilinkcustomer","MCB":"com.mcb.mcblive","MCB Islamic":"com.mcbislamicbank.mobileapp",
             "Meezan Bank":"invo8.meezan.mb","National Bank of Pakistan":"com.paysys.nbpdigital","Samba Bank":"com.ceesolutions.samba",
             "Standard Chartered":"com.scb.pk.bmw","Silk Bank":"com.goodbarber.silkbankmobile","Sindh Bank":"com.SindhBank.MobileBanking",
             "Soneri Bank":"com.p3.soneridigital","Summit Bank":"com.avanza.ambitwizsummit","UBL": "app.com.brd"}

option = st.sidebar.selectbox("Select Bank", ("Choose","ABL","AlBaraka","Alfalah","Askari","Bank Al-Habib","Bank of Khyber","Bank of Punjab",
                        "Dubai Islamic Bank","EasyPaisa","Faysa Bank","First Women Bank","HBL","Habib Metropolitian Bank","JazzCash",
                        "MCB","MCB Islamic","Meezan Bank","National Bank of Pakistan","Samba Bank","Standard Chartered","Silk Bank","Sindh Bank",
                        "Soneri Bank","Summit Bank","UBL"), index=0,  on_change=None)
# st.write(bank_dict[option])

if option == "Choose":
    pass
else:
    app_review(bank_dict[option])


