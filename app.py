import streamlit as st
import joblib
import os

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Flower Prediction")
st.write("Predict the Iris flower species using Machine Learning.")

if not os.path.exists("iris_model.pkl"):
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)
    joblib.dump(model, "iris_model.pkl")

model = joblib.load("iris_model.pkl")

sl = st.number_input("Sepal Length (cm)", value=5.1)
sw = st.number_input("Sepal Width (cm)", value=3.5)
pl = st.number_input("Petal Length (cm)", value=1.4)
pw = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):
    result = model.predict([[sl, sw, pl, pw]])[0]
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success("Predicted Flower: " + flowers[result])