from matplotlib.pyplot import figimage
import streamlit as st
import plotly.express as px
import pandas as pd

st.title("Plotly ki madad sa graphs banana")
df = px.data.gapminder()
st.write(df.head())
st.write(df.columns)

st.write(df.describe())


year_option = df['year'].unique().tolist()

# year = st.selectbox("which year should we plot",year_option,0)
# df = df[df['year']==year]

fig=px.scatter(df,x='gdpPercap',y=  'lifeExp',size='pop',color='country',
                hover_name='country',log_x=True,size_max=55,range_x=[100,10000],
                range_y=[20,90],animation_frame='year',animation_group='country')
st.write(fig)
