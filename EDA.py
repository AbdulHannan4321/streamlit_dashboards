import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Title foe Webapp

st.markdown('''# __Exploratory Data Analysis Web Application__

This application is developed by Abdul Hannan''')

#File uploading

with st.sidebar.header("Upload your dataset (.csv"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    df=sns.load_dataset("titanic")
    st.sidebar.markdown("[Example CSV File](https://www.kaggle.com/datasets/kkhandekar/sulfur-dioxide-pollution)")

#profiling report for pandas
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv =pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr=ProfileReport(df,explorative=True)
    st.header("__Input Dataframe__")
    st.write(df)
    st.write("---")
    st.header("__Profiling Report with Pandas")
    st_profile_report(pr)
else:
    st.info("Waiting for CSV File")
    if st.button("Press to use example Data"):
        @st.cache
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                                columns=['age','banana','codanics','duck','elephant'])
            return a
        df=load_data()
        pr=ProfileReport(df,explorative=True)
        st.header("__Input DataFrame__")
        st.write(df)
        st.write("---")
        st.header("__Pandas Profiling Report__")
        st_profile_report(pr)