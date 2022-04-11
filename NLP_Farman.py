import streamlit as st
from google_play_scraper import app
import pandas as pd
import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
# import csv
#import panel as pl
import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import re
# import unicodedata
# import unidecode
import string



st.markdown('''# __Sentiment Analysis of Mobile Banking App__

This application is developed by __Abdul Hannan__''')


from google_play_scraper import Sort, reviews_all

def app_review(x):
    # pak_review = reviews_all("app.com.brd", sleep_milliseconds=0,lang="en",country="pk",sort = Sort.MOST_RELEVANT)
    pak_review = reviews_all(x, sleep_milliseconds=0,lang="en",country="pk",sort = Sort.MOST_RELEVANT)
    df = pd.DataFrame(np.array(pak_review), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    # st.write(df)
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

    st.write(df)
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


