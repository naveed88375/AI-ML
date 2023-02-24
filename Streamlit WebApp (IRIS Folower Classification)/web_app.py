#Import required packages
import joblib
import pandas as pd
from sklearn import datasets
import streamlit as st

#Title and description
st.write("""
# Iris Flower Classification
This webapp will classify **Iris Flowers** based on the provided features.
""")

#Sidebar title
st.sidebar.header('Flower features')

#Function to get user features using a slider
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

    #Store provided features in pandas dataframe
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

#Get user features
df = user_input_features()

#Load iris dataset to get flower names
iris = datasets.load_iris()

#Display given features
st.subheader('Provided features')
st.write(df)

#Load the trained classifier model
clf = joblib.load("./random_forest.joblib")

#Perform predictions using trained classifier
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

#Display class labels
st.subheader('Class labels')
st.write(iris.target_names)

#Present model predictions
st.subheader('Model Predictions')
st.write(iris.target_names[prediction])

#Print class probabilities
st.subheader('Class Probability')
st.write(prediction_proba)