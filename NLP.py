import streamlit as st
from google_play_scraper import app
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import panel as pl
import nltk as nlp
from nltk.tokenize import word_tokenize



st.markdown('''# __Sentiment Analysis of Mobile Banking App__

This application is developed by __Abdul Hannan__''')


from google_play_scraper import Sort, reviews_all

def app_review(x):
    # pak_review = reviews_all("app.com.brd", sleep_milliseconds=0,lang="en",country="pk",sort = Sort.MOST_RELEVANT)
    pak_review = reviews_all(x, sleep_milliseconds=0,lang="en",country="pk",sort = Sort.MOST_RELEVANT)
    df = pd.DataFrame(np.array(pak_review), columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    # st.write(df.head())

review=.selectbox("Select Bank?", options=[""])

