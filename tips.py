import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# mse = mean_squared_error()
# mae = mean_absolute_error()
# r2 = r2_score()

header =st.container()
data_set = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Tips kay Data ki App")
    st.text("In this project we will use the tips data")

with data_set:
    st.header("Tip kab milay ge")
    # import dataset
    df = sns.load_dataset("tips")
    df = df.dropna()
    st.write(df.head())

    st.bar_chart(df['sex'].value_counts())
    st.bar_chart(df['day'].value_counts())

    st.bar_chart(df['total_bill'].sample(50))
    st.bar_chart(df['tip'].sample(50))

with features:
    st.header("These are the features of the app.")
    st.markdown("1. Different Features")
with model_training:
    st.header("Waiter ko tip milay ge ya nahi?")
    Input,display = st.columns(2)
    # Input.slider("How many People?",min_value=10,max_value=100,value=20,step=5)
    max_depth=Input.slider("How much the total bill?",min_value=10,max_value=100,value=20,step=5)

# n_estimator
n_estimator = Input.selectbox("How many tree would be there?", options=[50,100,200,300,"No Limit"])



# input_feature = Input.text_input("Which feature we should use")
input_feature = Input.selectbox("Which feature we should use",options=["total_bill","size","tip"])


#machine Learning Model

model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimator)
if n_estimator=="No Limit":
    model=RandomForestRegressor(max_depth=max_depth)
else:
    model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimator)
#define X and y
X = df[[input_feature]]
y = df[["tip"]]

#model fit
model.fit(X,y)
pred = model.predict(y)

#Display Metrics
display.subheader("Mean Absolute error of the model is:  ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean Squared error of the model is:  ")
display.write(mean_squared_error(y,pred))
display.subheader("R2 Score of the model is:  ")
display.write(r2_score(y,pred))