import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

# Page Config
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements below to predict the species.")

# Inputs
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# Prediction
if st.button("Predict Species"):
    
    features = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(features)[0]

    st.success(f"Predicted Species: {prediction}")