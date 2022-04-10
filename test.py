from turtle import color
import seaborn as sns
import streamlit as st
st.header("ABC")
st.text("Hello")
df = sns.load_dataset("iris")
st.write(df.head(10))
st.bar_chart(df["sepal_length"])
st.bar_chart(df["sepal_width"])